"""Pipeline orchestration (docs/ARCHITECTURE.md section 4b): the contract used by the CLI, Colab and tests.

Run dir layout::

    config.yaml        resolved config (driftlab.config.dump_yaml)
    plan.yaml          frozen copy of the analysis plan (written once; replaced only with allow_config_change)
    provenance.json    provenance of the run's first open (also in runs.provenance_json)
    store.sqlite       experiment store
    shards/            immutable shard log (default; Colab: a Drive folder via ``shard_dir``)
    exports/analysis/  analysis bundle;  exports/tables/  T1-T10 (md/csv/tex)

Stages (``STAGES``) are idempotent and resumable: each computes planned - done from the store and does only
the rest, so re-running a finished run makes no model calls. ``max_minutes`` is a soft deadline checked by the
engine between chunks; on expiry ``run()`` returns ``{"status": "budget_exhausted", ...}`` with everything
finished so far committed. Restore (Colab): put a ``Store.backup_to`` snapshot at ``<run_dir>/store.sqlite``
and point ``shard_dir`` at the shard log; ``open()`` replays newer shards BEFORE creating the shard writer.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from driftlab import data as data_mod
from driftlab.analysis.cube import Trajectory
from driftlab.audit import run_audit
from driftlab.backends import make_backend, resolve_kind
from driftlab.backends.base import Backend, GenRequest
from driftlab.config import AnalysisPlan, ExperimentConfig, dump_yaml, load_config, load_plan
from driftlab.data import DEV_SPLIT, EVAL_SPLIT, DevSplit, EvalSplit
from driftlab.engine import BudgetExhausted, CellTask, GenerationEngine, decoding_json
from driftlab.estimate import audit_slot_count
from driftlab.keys import engine_fingerprint, sha256_text
from driftlab.planning import (
    MatrixGroup,
    audit_slots,
    cell_tuple,
    check_audit_config,
    n_audit_items,
    plan_matrix,
    planned_cell_counts,
    planned_cell_keys,
)
from driftlab.provenance import collect_provenance
from driftlab.scoring import score_all, scoring_progress
from driftlab.store.shards import ShardWriter, replay_shards
from driftlab.store.store import CellConflict, Store
from driftlab.trajectory import (
    TrajectoryIncomplete,
    run_trajectory,
    trajectories_from_store,
    trajectory_progress,
)

STAGES: tuple[str, ...] = ("data", "trajectory", "matrix", "score", "audit", "analyze")
# Stages whose completion makes a run "complete" (analyze is a derived, re-runnable product).
DATA_STAGES: tuple[str, ...] = ("data", "trajectory", "matrix", "score", "audit")
STORE_FILE = "store.sqlite"
CONFIG_FILE = "config.yaml"
PLAN_FILE = "plan.yaml"
PROVENANCE_FILE = "provenance.json"
SHARD_DIRNAME = "shards"
DEFAULT_CHUNK_SIZE = 512


def _atomic_write(path: Path, data: str | bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if isinstance(data, str):
        tmp.write_text(data, encoding="utf-8")
    else:
        tmp.write_bytes(data)
    os.replace(tmp, path)


class Pipeline:
    """Orchestrates one run dir: data -> trajectory -> matrix -> score -> audit -> analyze."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        run_dir: str | Path,
        backend: Backend | None = None,
        *,
        run_id: str | None = None,
        shard_dir: str | Path | None = None,
        allow_engine_change: bool = False,
        allow_config_change: bool = False,
        checkpoint_hook: Callable[[str], None] | None = None,
        log: Callable[[str], None] = print,
        fail_after_chunks: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_id = run_id or cfg.run.name
        self.shard_dir = Path(shard_dir) if shard_dir is not None else self.run_dir / SHARD_DIRNAME
        self.allow_engine_change = allow_engine_change
        self.allow_config_change = allow_config_change
        self.checkpoint_hook = checkpoint_hook
        self.log = log
        self.fail_after_chunks = fail_after_chunks  # None -> DRIFTLAB_FAIL_AFTER_CHUNKS (engine default)
        self.backend: Backend | None = backend
        self._owns_backend = backend is None
        self.store: Store | None = None
        self.shard_writer: ShardWriter | None = None
        self.engine_fp: str | None = None
        self.plan: AnalysisPlan | None = None
        self._dev: DevSplit | None = None
        self._eval: EvalSplit | None = None
        self._trajs: dict[int, Trajectory] | None = None

    # ------------------------------------------------------------------ lifecycle
    @property
    def store_path(self) -> Path:
        return self.run_dir / STORE_FILE

    def open(self) -> None:
        """Store (+ shard replay), backend, provenance and ``ensure_run``; then write the run-dir files.

        Raises :class:`~driftlab.store.ConfigMismatch` / :class:`~driftlab.store.EngineMismatch` when the run
        dir holds a different config / engine (unless allowed); nothing is overwritten in that case.

        ``plan.yaml`` is the run's frozen copy of the pre-registered analysis plan: it is written when the run
        dir has none and is never replaced by a later edit of ``cfg.analysis_plan`` (the run keeps analysing
        with the plan it was started with, as ``analyze`` does) unless ``allow_config_change`` (then the
        change is recorded by ``ensure_run`` as ``meta['plan_change:<n>']``).
        """
        if self.store is not None:
            return
        cfg = self.cfg
        plan_src = cfg.resolve_path(cfg.analysis_plan)
        if not plan_src.is_file():
            raise FileNotFoundError(f"analysis plan not found: {plan_src}")
        plan_dst = self.run_dir / PLAN_FILE
        plan_bytes = plan_src.read_bytes()
        write_plan = True
        if plan_dst.is_file() and plan_dst.read_bytes() != plan_bytes:
            frozen, source = load_plan(plan_dst), load_plan(plan_src)
            same = frozen.plan_hash() == source.plan_hash()
            if self.allow_config_change:
                self.log(
                    f"WARNING: {plan_dst} differs from {plan_src}; replacing the run's frozen plan "
                    f"(allow_config_change{'' if same else '; recorded as a plan change'})"
                )
            else:
                write_plan = False
                self.log(
                    f"WARNING: {plan_src} differs from the run's frozen plan {plan_dst}"
                    f"{' (same parsed plan)' if same else ''}; keeping the frozen copy "
                    "(pass allow_config_change to replace it)"
                )
        self.plan = load_plan(plan_src if write_plan else plan_dst)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        store = Store(self.store_path)
        created = False
        try:
            if self.shard_dir.is_dir():  # restore: replay BEFORE creating the writer (else ShardLogAhead)
                n = replay_shards(store, self.shard_dir)
                if n:
                    self.log(f"[open] replayed {n} row(s) from {self.shard_dir}")
            if self.backend is None:
                key = data_mod.answer_key(cfg) if resolve_kind(cfg.backend.kind) == "mock" else None
                self.backend = make_backend(cfg, answer_key=key)
                created = True
            self.engine_fp = engine_fingerprint(self.backend.engine_info())
            existing = store.get_run(self.run_id)
            prov = None  # provenance_json is kept from the run's first open
            if existing is None or not existing.get("provenance_json"):
                prov = collect_provenance(cfg, self.backend)
            row = store.ensure_run(
                self.run_id,
                config_json=cfg.model_dump(mode="json"),
                config_hash=cfg.config_hash(),
                plan_hash=self.plan.plan_hash(),
                provenance_json=prov,
                engine_fp=self.engine_fp,
                synthetic=bool(getattr(self.backend, "synthetic", False)),
                allow_config_change=self.allow_config_change,
                allow_engine_change=self.allow_engine_change,
            )
            self.shard_writer = ShardWriter.for_store(store, self.shard_dir)
        except BaseException:
            store.close()
            if created and self.backend is not None:
                self.backend.close()
                self.backend = None
            raise
        self.store = store
        _atomic_write(self.run_dir / CONFIG_FILE, dump_yaml(cfg))
        if write_plan:
            _atomic_write(plan_dst, plan_bytes)
        prov_path = self.run_dir / PROVENANCE_FILE
        if not prov_path.exists():
            stored = row.get("provenance_json")
            record = json.loads(stored) if stored else (prov or collect_provenance(cfg, self.backend))
            _atomic_write(prov_path, json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")

    def close(self) -> None:
        if self.store is not None:
            self.store.close()
            self.store = None
        if self._owns_backend and self.backend is not None:
            self.backend.close()
            self.backend = None
        self.shard_writer = None

    def __enter__(self) -> Pipeline:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _store(self) -> Store:
        self.open()
        assert self.store is not None
        return self.store

    # ------------------------------------------------------------------ inputs
    def dev_split(self) -> DevSplit:
        if self._dev is None:
            self._dev = data_mod.load_dev(self.cfg)
        return self._dev

    def eval_split(self) -> EvalSplit:
        if self._eval is None:
            self._eval = data_mod.load_eval(self.cfg)
        return self._eval

    def trajectories(self) -> dict[int, Trajectory]:
        """Completed trajectories (from this process or rebuilt from the store); RuntimeError if Phase A is
        incomplete."""
        if self._trajs is None:
            try:
                self._trajs = trajectories_from_store(self.cfg, self._store(), self.run_id)
            except TrajectoryIncomplete as e:
                raise RuntimeError(f"run the 'trajectory' stage first: {e}") from e
        return self._trajs

    def _engine(self, deadline: float | None) -> GenerationEngine:
        assert self.backend is not None
        kind = getattr(self.backend, "kind", "")
        chunk = int(self.cfg.matrix.chunk_size.get(kind, DEFAULT_CHUNK_SIZE))
        return GenerationEngine(
            self.backend,
            self._store(),
            self.run_id,
            chunk_size=chunk,
            shard_writer=self.shard_writer,
            deadline=deadline,
            fail_after_chunks=self.fail_after_chunks,
            engine_fp=self.engine_fp,
        )

    def _hook(self, stage: str) -> None:
        if self.checkpoint_hook is not None:
            self.checkpoint_hook(stage)

    @staticmethod
    def _stages(stages: Sequence[str] | str | None) -> list[str]:
        if stages is None:
            return list(STAGES)
        wanted = [stages] if isinstance(stages, str) else list(stages)
        unknown = [s for s in wanted if s not in STAGES]
        if unknown:
            raise ValueError(f"unknown stage(s) {unknown}; expected some of {STAGES}")
        return [s for s in STAGES if s in wanted]

    # ------------------------------------------------------------------ run
    def run(self, stages: Sequence[str] | None = None, max_minutes: float | None = None) -> dict:
        """Run ``stages`` (default all, always in ``STAGES`` order). Returns ``{"status": "ok" |
        "budget_exhausted", "run_id", "stages": {stage: summary}, "executed", "requested", "progress":
        status(), ...}``.

        Stage names and (when the audit stage will run) the audit config are validated BEFORE the run dir is
        touched, so a misconfigured audit fails at once instead of after hours of trajectory / matrix work.
        """
        wanted = self._stages(stages)
        if "audit" in wanted:
            check_audit_config(self.cfg)
        store = self._store()
        deadline = None if max_minutes is None else time.monotonic() + 60.0 * float(max_minutes)
        engine = self._engine(deadline)
        results: dict[str, Any] = {}
        current = ""
        try:
            for name in wanted:
                current = name
                results[name] = getattr(self, f"_stage_{name}")(engine)
                self._hook(name)
        except BudgetExhausted as e:
            store.set_stage_status(self.run_id, current, None, None, None, "budget_exhausted")
            self.log(
                f"[{current}] time budget exhausted ({e}); finished work is committed, re-run to continue"
            )
            progress = self.status()
            return {
                "status": "budget_exhausted",
                "stage": current,
                "run_id": self.run_id,
                "stages": results,
                "executed": engine.n_executed,
                "requested": engine.n_requested,
                "complete": progress["complete"],
                "synthetic": progress["synthetic"],
                "message": str(e),
                "progress": progress,
            }
        progress = self.status()
        return {
            "status": "ok",
            "run_id": self.run_id,
            "stages": results,
            "executed": engine.n_executed,
            "requested": engine.n_requested,
            "complete": progress["complete"],
            "synthetic": progress["synthetic"],
            "progress": progress,
        }

    # ------------------------------------------------------------------ stages
    def _stage_data(self, engine: GenerationEngine) -> dict:
        store, written = self._store(), 0
        for split, items in ((DEV_SPLIT, self.dev_split()), (EVAL_SPLIT, self.eval_split())):
            rows = [it.to_row() for it in items]
            stored = {int(r["idx"]): r for r in store.get_items(split)}
            same = all(
                r["idx"] in stored
                and all(str(stored[r["idx"]][c]) == str(r[c]) for c in ("question", "answer_text", "gold"))
                for r in rows
            )
            if not same:  # put_items raises ItemConflict on changed content
                store.put_items(split, rows, shard_writer=self.shard_writer, stage="items")
                written += len(rows)
        n = len(self.dev_split()) + len(self.eval_split())
        store.set_stage_status(self.run_id, "data", None, n, n, "complete")
        return {"dev": len(self.dev_split()), "eval": len(self.eval_split()), "written": written}

    def _stage_trajectory(self, engine: GenerationEngine) -> dict:
        store, cfg = self._store(), self.cfg
        n0 = engine.n_executed
        trajs = run_trajectory(
            cfg,
            engine,
            store,
            self.run_id,
            self.dev_split(),
            cfg.run.seeds,
            log=self.log,
            checkpoint=lambda: self._hook("trajectory"),
        )
        self._trajs = trajs
        n = len(cfg.run.seeds) * cfg.run.rounds
        store.set_stage_status(self.run_id, "trajectory", None, n, n, "complete")
        return {
            "executed": engine.n_executed - n0,
            "final_incumbent": {s: t.incumbent_after(t.R) for s, t in trajs.items()},
            "n_advanced": {s: sum(t.advanced) for s, t in trajs.items()},
            "n_fallback": {s: sum(t.is_fallback) for s, t in trajs.items()},
        }

    def _stored_requests(self, store: Store) -> dict[tuple, tuple]:
        """``{cell tuple: engine-independent request identity}`` of the run's stored eval cells whose
        generation is present (system / user hashes, decoding JSON, seed if sampling, nonce)."""
        rows = store.query(
            "SELECT c.seed, c.split, c.decoding_id, c.slot, c.draw_kind, c.draw, c.item_idx, g.system_hash, "
            "g.user_hash, g.decoding_json, g.seed AS gen_seed, g.nonce FROM cells c "
            "JOIN generations g ON g.gen_key = c.gen_key WHERE c.run_id = ? AND c.split = ?",
            (self.run_id, EVAL_SPLIT),
        )
        return {
            (r["seed"], r["split"], r["decoding_id"], r["slot"], r["draw_kind"], r["draw"], r["item_idx"]): (
                r["system_hash"],
                r["user_hash"],
                r["decoding_json"],
                r["gen_seed"],
                r["nonce"],
            )
            for r in rows
        }

    def _stage_matrix(self, engine: GenerationEngine) -> dict:
        """Run every planned cell that is not stored yet.

        Stored cells are checked against the plan first: cells are first-write-wins (ARCHITECTURE section 5),
        so a stored cell whose request differs from the planned one (prompt, user turn, decoding, seed or
        nonce: the matrix plan changed, e.g. under ``allow_config_change``) raises
        :class:`~driftlab.store.store.CellConflict` before any generation instead of silently keeping the old
        plan's generation. The check ignores the engine, so a deliberate ``allow_engine_change`` resumes.
        """
        store = self._store()
        groups = plan_matrix(self.cfg, self.plan, self.trajectories(), self.eval_split(), run_id=self.run_id)
        stored = self._stored_requests(store)
        hashes: dict[str, str] = {}

        def ident(req: GenRequest) -> tuple:
            for s in (req.system, req.user):
                if s not in hashes:
                    hashes[s] = sha256_text(s)
            seed = None if req.decoding.is_greedy else req.seed
            return (hashes[req.system], hashes[req.user], decoding_json(req), seed, req.nonce)

        todos: list[tuple[MatrixGroup, list[CellTask]]] = []
        conflicts: list[tuple] = []
        for g in groups:
            todo = []
            for t in g.tasks:
                assert t.cell is not None
                have = stored.get(cell_tuple(t.cell))
                if have is None:
                    todo.append(t)
                elif have != ident(t.request):
                    conflicts.append(cell_tuple(t.cell))
            todos.append((g, todo))
        if conflicts:
            raise CellConflict(
                f"{len(conflicts)} stored matrix cell(s) were generated for a different request than the "
                f"current plan, e.g. {conflicts[0]} (nonce, seed, prompt, user turn or decoding changed). Cells "
                "are first-write-wins: a changed matrix plan needs a new run dir."
            )
        n0, n_groups, n_cells = engine.n_executed, 0, 0
        for g, todo in todos:
            if not todo:
                continue
            engine.run(todo, stage="matrix", purpose=g.purpose, seed=g.seed, round_=g.round)
            n_groups += 1
            n_cells += len(todo)
            self.log(f"[matrix] seed {g.seed} {g.decoding_id} {g.draw_kind} {g.draw}: {len(todo)} cell(s)")
            self._hook("matrix")
        planned = sum(len(g) for g in groups)
        store.set_stage_status(self.run_id, "matrix", None, planned, planned, "complete")
        return {
            "planned": planned,
            "groups_run": n_groups,
            "cells_written": n_cells,
            "executed": engine.n_executed - n0,
        }

    def _stage_score(self, engine: GenerationEngine) -> dict:
        self._stage_data(engine)  # scores need the items (gold); no-op when stored
        store = self._store()
        written = score_all(store, self.run_id, shard_writer=self.shard_writer)
        prog = scoring_progress(store, self.run_id)
        planned = sum(v["planned"] for v in prog.values())
        left = sum(v["unscored"] for v in prog.values())
        store.set_stage_status(
            self.run_id, "score", None, planned, planned - left, "complete" if not left else "partial"
        )
        self.log(f"[score] wrote {written} score row(s)")
        return {"written": written}

    def _audit_planned(self, trajs: dict[int, Trajectory]) -> set[tuple[str, int, int]]:
        """``(decoding, slot, repeat)`` audit_results rows the config asks for (empty without audited items)."""
        a = self.cfg.audit
        check_audit_config(self.cfg)
        if a.seed not in trajs:
            raise ValueError(f"audit.seed {a.seed} is not a run seed {sorted(trajs)}")
        if n_audit_items(self.cfg) == 0:
            return set()
        return {
            (d, k, r)
            for k in audit_slots(self.cfg, trajs[a.seed])
            for d in a.decodings
            for r in range(a.repeats)
        }

    def _audit_done(self, store: Store) -> set[tuple[str, int, int]]:
        return {
            (r["decoding_id"], int(r["slot"]), int(r["repeat"])) for r in store.get_audit_results(self.run_id)
        }

    def _stage_audit(self, engine: GenerationEngine) -> dict:
        if not self.cfg.audit.enabled:
            return {"skipped": "audit disabled"}
        store, trajs = self._store(), self.trajectories()
        planned = self._audit_planned(trajs)
        if planned <= self._audit_done(store):
            store.set_stage_status(self.run_id, "audit", None, len(planned), len(planned), "complete")
            return {"skipped": "complete", "rows": len(planned)}
        rows = run_audit(self.cfg, engine, store, self.run_id, trajs, self.eval_split(), log=self.log)
        scored = score_all(store, self.run_id, shard_writer=self.shard_writer)  # score the audit generations
        store.set_stage_status(self.run_id, "audit", None, len(planned), len(rows), "complete")
        return {"rows": len(rows), "scored": scored}

    def _stage_analyze(self, engine: GenerationEngine) -> dict:
        store = self._store()
        try:
            from driftlab.analysis.bundle import analyze
        except ImportError as e:
            self.log(f"WARNING: analyze stage skipped (analysis module unavailable: {e})")
            store.set_stage_status(self.run_id, "analyze", None, 1, 0, "skipped")
            return {"skipped": f"ImportError: {e}"}
        bundle = analyze(self.run_dir, log=self.log)
        out: dict[str, Any] = {"bundle_dir": str(self.run_dir / "exports" / "analysis")}
        try:
            from driftlab.reporting.tables import write_tables
        except ImportError as e:  # bundle built; tables can be written later (`driftlab tables`)
            self.log(f"WARNING: tables skipped (reporting module unavailable: {e})")
            store.set_stage_status(self.run_id, "analyze", None, 1, 0, "partial")
            return {**out, "tables_skipped": f"ImportError: {e}"}
        tables_dir = self.run_dir / "exports" / "tables"
        try:
            write_tables(bundle, tables_dir)
        except ValueError as e:  # e.g. a SYNTHETIC run dir under a 'results' directory: refused by policy
            self.log(f"WARNING: tables not written: {e}")
            store.set_stage_status(self.run_id, "analyze", None, 1, 0, "partial")
            return {**out, "tables_skipped": f"ValueError: {e}"}
        store.set_stage_status(self.run_id, "analyze", None, 1, 1, "complete")
        return {**out, "tables_dir": str(tables_dir)}

    # ------------------------------------------------------------------ status
    @contextmanager
    def _reader(self) -> Iterator[Store]:
        """The open store, else a temporary READ-ONLY view of the run dir's store (an empty in-memory store
        when there is none yet): ``status()`` on a closed pipeline never replays shards, writes the run row,
        rewrites run-dir files, builds a backend or quarantines a concurrent writer's in-flight shard."""
        if self.store is not None:
            yield self.store
            return
        view = Store(self.store_path, read_only=True) if self.store_path.is_file() else Store(":memory:")
        try:
            yield view
        finally:
            view.close()

    def status(self) -> dict:
        """Per stage ``{"planned", "done", "complete", ...}`` derived from the stored data (not from flags).

        ``complete`` covers every model-call stage plus scoring (data, trajectory, matrix, score, audit if
        enabled); ``stages["analyze"]["complete"]`` says whether the analysis bundle exists. Read-only: a
        closed pipeline is not opened (see :meth:`_reader`).
        """
        with self._reader() as store:
            return self._status(store)

    def _status(self, store: Store) -> dict:
        cfg = self.cfg
        seeds, R = list(cfg.run.seeds), cfg.run.rounds
        stages: dict[str, dict[str, Any]] = {}

        n_items = 0
        for split, n in ((DEV_SPLIT, cfg.data.dev.n), (EVAL_SPLIT, cfg.data.eval.n)):
            q = store.query("SELECT COUNT(*) AS n FROM items WHERE split = ? AND idx < ?", (split, n))
            n_items += int(q[0]["n"])
        planned = cfg.data.dev.n + cfg.data.eval.n
        stages["data"] = {"planned": planned, "done": n_items, "complete": n_items >= planned}

        per_seed = trajectory_progress(store, self.run_id, seeds)
        done = sum(min(v, R) for v in per_seed.values())
        stages["trajectory"] = {
            "planned": len(seeds) * R,
            "done": done,
            "complete": done >= len(seeds) * R,
            "per_seed": per_seed,
        }

        keys = planned_cell_keys(cfg)
        existing = store.existing_cell_keys(self.run_id, split=EVAL_SPLIT)
        done = len(keys & existing)
        stages["matrix"] = {
            "planned": len(keys),
            "done": done,
            "complete": stages["trajectory"]["complete"] and done == len(keys),
            "by_purpose": planned_cell_counts(cfg),
        }

        prog = scoring_progress(store, self.run_id)
        planned = sum(v["planned"] for v in prog.values())
        left = sum(v["unscored"] for v in prog.values())
        stages["score"] = {
            "planned": planned,
            "done": planned - left,
            "unscored": left,
            "complete": stages["matrix"]["complete"] and left == 0,
            "by_extractor": prog,
        }

        if not cfg.audit.enabled:
            stages["audit"] = {"enabled": False, "planned": 0, "done": 0, "complete": True}
        else:
            try:
                trajs = self._trajs or trajectories_from_store(cfg, store, self.run_id)
            except TrajectoryIncomplete:
                trajs = None
            try:
                check_audit_config(cfg)
                want = None if trajs is None else self._audit_planned(trajs)
            except ValueError as e:
                stages["audit"] = {
                    "enabled": True,
                    "planned": 0,
                    "done": 0,
                    "complete": False,
                    "error": str(e),
                }
            else:
                if want is None:  # slots unknown until Phase A is complete: config-level estimate
                    n = audit_slot_count(cfg) * len(cfg.audit.decodings) * cfg.audit.repeats
                    stages["audit"] = {
                        "enabled": True,
                        "planned": n if n_audit_items(cfg) else 0,
                        "done": 0,
                        "complete": False,
                    }
                else:
                    got = len(want & self._audit_done(store))
                    stages["audit"] = {
                        "enabled": True,
                        "planned": len(want),
                        "done": got,
                        "complete": got == len(want),
                    }

        bundle = self.run_dir / "exports" / "analysis" / "bundle.json"
        stages["analyze"] = {"planned": 1, "done": int(bundle.exists()), "complete": bundle.exists()}

        row = store.get_run(self.run_id)
        led = store.query(
            "SELECT COALESCE(SUM(n_requested), 0) AS req, COALESCE(SUM(n_executed), 0) AS exe FROM ledger "
            "WHERE run_id = ? AND stage NOT LIKE '%:error'",
            (self.run_id,),
        )[0]
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "stages": stages,
            "complete": all(stages[s]["complete"] for s in DATA_STAGES),
            "synthetic": bool(row["synthetic"]) if row is not None else False,
            "requested": int(led["req"]),
            "executed": int(led["exe"]),
        }


def run_experiment(
    config_path: str | Path,
    run_dir: str | Path,
    *,
    overrides: list[str] | None = None,
    stages: Sequence[str] | None = None,
    max_minutes: float | None = None,
    backend: Backend | None = None,
) -> dict:
    """Load a config, open ``run_dir`` and run ``stages``; returns :meth:`Pipeline.run`'s dict + ``run_dir``."""
    cfg = load_config(config_path, overrides)
    with Pipeline(cfg, run_dir, backend) as p:
        result = p.run(stages, max_minutes)
    result["run_dir"] = str(run_dir)
    return result


__all__ = ["DATA_STAGES", "STAGES", "Pipeline", "run_experiment"]
