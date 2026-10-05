"""Cache-aware, chunked, ledgered, time-budgeted generation engine (docs/ARCHITECTURE.md section 5).

``GenerationEngine.run(tasks)``:

1. render each request with the backend's chat template and compute its content-addressed ``gen_key``
   (engine fingerprint, model revision, rendered prompt, explicit decoding, seed if sampling, nonce);
2. dedupe identical keys within the call and look up generations already in the store;
3. generate the missing keys in order of first appearance, in chunks of ``chunk_size``;
4. per chunk, ONE store transaction writes the new generations, every cell whose generation is now
   available (cache-hit cells go with the first chunk, or a final zero-generation chunk) and one ledger row
   (requested = logical tasks, executed = cache misses; backend errors get an extra ``"<stage>:error"``
   row), plus one immutable shard file if configured.

Budget checks (deadline, injected failure) run *between* chunks, after the previous chunk committed, so a
crash loses at most the chunk in flight and a re-run of the same tasks converges to the same store. They
run only before chunks that call the backend and count only such chunks: an all-cache-hit call is free and
always completes, so re-submitting finished work after a crash can never starve progress.

Integrity guards (checked before any model call): a sampling request must carry an explicit seed (an
unseeded sample would be cached and silently reused by every "independent" draw); a cell's
``decoding_id`` must match its request; one call may not map a cell to two generations; a cell already in
the store with a different ``gen_key`` raises :class:`~driftlab.store.store.CellConflict` (cells are
first-write-wins, so the new generation would be paid for but never reach the cube); and the engine
fingerprint must match the run's stored ``engine_fp``. A synthetic backend (mock) always marks the run
``synthetic`` (sticky), even if the caller forgot to.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from driftlab.backends.base import Backend, GenRequest, GenResult
from driftlab.keys import engine_fingerprint, gen_key, sha256_text
from driftlab.store.shards import ShardWriter
from driftlab.store.store import CellConflict, EngineMismatch, Store, utc_now

FAIL_AFTER_ENV = "DRIFTLAB_FAIL_AFTER_CHUNKS"


@dataclass(frozen=True)
class CellKey:
    """Logical generation event: one prompt (slot) answering one item at one draw."""

    run_id: str
    seed: int
    split: str
    decoding_id: str
    slot: int
    draw_kind: str  # "round" | "gt" | "audit"
    draw: int
    item_idx: int


@dataclass(frozen=True)
class CellTask:
    """A request plus the cell it fills (``cell=None`` for proposer calls: stored, but no cell row)."""

    cell: CellKey | None
    request: GenRequest
    physical: bool = False


@dataclass
class GenRecord:
    gen_key: str
    text: str
    finish_reason: str
    cached: bool  # served from the store or from an identical earlier task of the same call
    n_completion_tokens: int


class BudgetExhausted(Exception):
    """The time budget (``deadline``) ran out between chunks; everything committed so far is kept."""


class InjectedFailure(Exception):
    """Deliberate crash after N committed chunks (``fail_after_chunks`` / DRIFTLAB_FAIL_AFTER_CHUNKS)."""


class GenerationError(RuntimeError):
    """The backend returned ``finish_reason == "error"``; such outputs are never cached (the rest of the
    chunk is committed and the failed calls are logged in a ledger row with stage ``"<stage>:error"``)."""


def decoding_json(request: GenRequest) -> str:
    d = request.decoding
    return json.dumps({"id": d.id, **d.params()}, sort_keys=True, separators=(",", ":"))


class GenerationEngine:
    def __init__(
        self,
        backend: Backend,
        store: Store,
        run_id: str,
        *,
        chunk_size: int = 512,
        shard_writer: ShardWriter | None = None,
        deadline: float | None = None,
        fail_after_chunks: int | None = None,
        engine_fp: str | None = None,
        on_progress: Callable[[dict], None] | None = None,
    ) -> None:
        if int(chunk_size) < 1:
            raise ValueError("chunk_size must be >= 1")
        self.backend = backend
        self.store = store
        self.run_id = run_id
        self.chunk_size = int(chunk_size)
        self.shard_writer = shard_writer
        self.deadline = deadline  # time.monotonic() value
        if fail_after_chunks is None:
            env = os.environ.get(FAIL_AFTER_ENV, "").strip()
            fail_after_chunks = int(env) if env else None
        self.fail_after_chunks = fail_after_chunks
        self.engine_fp = engine_fp or engine_fingerprint(backend.engine_info())
        self.on_progress = on_progress
        # lifetime counters of this engine instance
        self.chunks_committed = 0  # every committed chunk (incl. zero-generation cache-hit chunks)
        self.chunks_executed = 0  # committed chunks that called the backend (budget checks count these)
        self.n_requested = 0
        self.n_executed = 0

    # ------------------------------------------------------------------ keys
    def _key(self, request: GenRequest, rendered_sha: str) -> str:
        return gen_key(
            rendered_sha=rendered_sha,
            decoding=request.decoding,
            seed=request.seed,
            nonce=request.nonce,
            model_id=self.backend.model_id,
            model_revision=self.backend.model_revision,
            engine_fp=self.engine_fp,
        )

    def key_for(self, request: GenRequest) -> str:
        """The ``gen_key`` this engine would assign to ``request``."""
        return self._key(request, sha256_text(self.backend.render(request.system, request.user)))

    # ------------------------------------------------------------------ guards
    def _check_run(self) -> None:
        """Engine fingerprint must match the run's; a synthetic backend always marks the run synthetic."""
        row = self.store.get_run(self.run_id)
        if row is None:
            return
        stored = row["engine_fp"]
        if stored is not None and stored != self.engine_fp:
            raise EngineMismatch(
                f"run {self.run_id!r} is recorded with engine {stored}, this engine is {self.engine_fp} "
                "(Store.ensure_run(..., allow_engine_change=True) records a deliberate change)"
            )
        if getattr(self.backend, "synthetic", False) and not row["synthetic"]:
            self.store.mark_synthetic(self.run_id)

    @staticmethod
    def _check_task(i: int, task: CellTask, run_id: str) -> None:
        req, c = task.request, task.cell
        if not req.decoding.is_greedy and req.seed is None:
            raise ValueError(
                f"task {i}: sampling request (decoding {req.decoding.id!r}) without a seed; derive one via "
                "driftlab.keys (an unseeded sample would be cached and reused by every draw)"
            )
        if c is None:
            return
        if c.run_id != run_id:
            raise ValueError(f"task {i} belongs to run {c.run_id!r}, engine runs {run_id!r}")
        if c.decoding_id != req.decoding.id:
            raise ValueError(
                f"task {i}: cell decoding {c.decoding_id!r} != request decoding {req.decoding.id!r}"
            )

    def _check_cells(self, cell_keys: dict[tuple, str]) -> None:
        existing = self.store.lookup_cells(self.run_id, cell_keys)
        bad = sorted(ck for ck, row in existing.items() if row["gen_key"] != cell_keys[ck])
        if bad:
            ck = bad[0]
            raise CellConflict(
                f"{len(bad)} cell(s) already stored with a different gen_key, e.g. {ck}: stored "
                f"{existing[ck]['gen_key'][:12]}, requested {cell_keys[ck][:12]}. The plan changed (nonce, seed, "
                "prompt, decoding, engine); cells are first-write-wins, so use a new run dir."
            )

    # ------------------------------------------------------------------ budget
    def _check_budget(self) -> None:
        """Called before every chunk that needs the backend (zero-generation chunks are free and always
        run, so re-submitting completed work can never starve progress)."""
        n = self.chunks_executed
        if self.fail_after_chunks is not None and n >= self.fail_after_chunks:
            raise InjectedFailure(f"injected failure after {n} committed generation chunk(s)")
        if self.deadline is not None and n > 0 and time.monotonic() >= self.deadline:
            raise BudgetExhausted(f"time budget exhausted after {n} committed generation chunk(s)")

    # ------------------------------------------------------------------ rows
    def _generation_row(
        self, key: str, rendered_sha: str, req: GenRequest, res: GenResult, batch_id: str
    ) -> dict[str, Any]:
        return {
            "gen_key": key,
            "engine_fp": self.engine_fp,
            "model_id": self.backend.model_id,
            "model_revision": self.backend.model_revision,
            "rendered_sha": rendered_sha,
            "system_hash": sha256_text(req.system),
            "user_hash": sha256_text(req.user),
            "decoding_json": decoding_json(req),
            "seed": None if req.decoding.is_greedy else req.seed,
            "nonce": req.nonce,
            "response": res.text,
            "finish_reason": res.finish_reason,
            "n_prompt_tokens": int(res.n_prompt_tokens),
            "n_completion_tokens": int(res.n_completion_tokens),
            "latency_ms": float(res.latency_ms),
            "batch_id": batch_id,
            "created_at": utc_now(),
        }

    def _cell_row(self, task: CellTask, key: str) -> dict[str, Any]:
        c = task.cell
        assert c is not None
        return {
            "run_id": c.run_id,
            "seed": c.seed,
            "split": c.split,
            "decoding_id": c.decoding_id,
            "slot": c.slot,
            "draw_kind": c.draw_kind,
            "draw": c.draw,
            "item_idx": c.item_idx,
            "gen_key": key,
            "physical": int(task.physical),
        }

    # ------------------------------------------------------------------ main entry point
    def run(
        self,
        tasks: Sequence[CellTask],
        *,
        stage: str,
        purpose: str,
        seed: int | None = None,
        round_: int | None = None,
    ) -> list[GenRecord]:
        """Serve every task (cache or backend) and commit chunk by chunk; returns records aligned with
        ``tasks``. Raises :class:`BudgetExhausted` / :class:`InjectedFailure` between chunks."""
        if not tasks:
            return []
        for i, t in enumerate(tasks):
            self._check_task(i, t, self.run_id)
        self._check_run()
        keys: list[str] = []
        first: dict[str, int] = {}  # key -> index of the first task with that key
        rendered: dict[str, str] = {}  # key -> rendered prompt sha
        render_cache: dict[tuple[str, str], str] = {}
        cell_keys: dict[tuple, str] = {}  # (seed, split, decoding_id, slot, draw_kind, draw, item) -> key
        for i, t in enumerate(tasks):
            req = t.request
            rs = render_cache.get((req.system, req.user))
            if rs is None:
                rs = sha256_text(self.backend.render(req.system, req.user))
                render_cache[(req.system, req.user)] = rs
            k = self._key(req, rs)
            keys.append(k)
            if k not in first:
                first[k] = i
                rendered[k] = rs
            c = t.cell
            if c is not None:
                ck = (c.seed, c.split, c.decoding_id, c.slot, c.draw_kind, c.draw, c.item_idx)
                if cell_keys.setdefault(ck, k) != k:
                    raise ValueError(f"task {i}: cell {ck} is requested twice with different generations")
        self._check_cells(cell_keys)

        known = self.store.lookup_generations(first)
        missing = [k for k in first if k not in known]
        chunks = [missing[i : i + self.chunk_size] for i in range(0, len(missing), self.chunk_size)] or [[]]
        chunk_of = {k: ci for ci, chunk in enumerate(chunks) for k in chunk}
        share: list[list[int]] = [[] for _ in chunks]  # task indices committed with each chunk
        for i, k in enumerate(keys):
            share[chunk_of.get(k, 0)].append(i)

        records: list[GenRecord | None] = [None] * len(tasks)
        done = 0
        for ci, chunk in enumerate(chunks):
            if chunk:
                self._check_budget()
            t0 = time.monotonic()
            gen_rows: list[dict[str, Any]] = []
            failed: dict[str, GenResult] = {}
            if chunk:
                reqs = [tasks[first[k]].request for k in chunk]
                results = self.backend.generate(reqs)
                if len(results) != len(reqs):
                    raise RuntimeError(f"backend returned {len(results)} results for {len(reqs)} requests")
                batch_id = f"{stage}:{ci}"
                for k, req, res in zip(chunk, reqs, results, strict=True):
                    if res.finish_reason == "error":
                        failed[k] = res
                        continue
                    row = self._generation_row(k, rendered[k], req, res, batch_id)
                    gen_rows.append(row)
                    known[k] = row
            fresh = {r["gen_key"] for r in gen_rows}
            cells: list[dict[str, Any]] = []
            n_req = 0
            for i in share[ci]:
                k = keys[i]
                if k in failed:
                    continue
                row = known[k]
                n_req += 1
                records[i] = GenRecord(
                    gen_key=k,
                    text=row["response"],
                    finish_reason=row["finish_reason"] or "",
                    cached=not (k in fresh and first[k] == i),
                    n_completion_tokens=int(row["n_completion_tokens"] or 0),
                )
                if tasks[i].cell is not None:
                    cells.append(self._cell_row(tasks[i], k))
            ledger = {
                "run_id": self.run_id,
                "seed": seed,
                "stage": stage,
                "purpose": purpose,
                "round": round_,
                "n_requested": n_req,
                "n_executed": len(gen_rows),
                "n_cache_hits": n_req - len(gen_rows),
                "prompt_tokens": sum(int(r["n_prompt_tokens"] or 0) for r in gen_rows),
                "completion_tokens": sum(int(r["n_completion_tokens"] or 0) for r in gen_rows),
                "wall_s": time.monotonic() - t0,
                "created_at": utc_now(),
            }
            ledgers = [ledger]
            if failed:  # paid-for model calls that produced no usable output: logged, never cached
                n_fail_req = sum(keys[i] in failed for i in share[ci])
                ledgers.append(
                    {
                        **ledger,
                        "stage": f"{stage}:error",
                        "n_requested": n_fail_req,
                        "n_executed": len(failed),
                        "n_cache_hits": n_fail_req - len(failed),
                        "prompt_tokens": sum(int(r.n_prompt_tokens or 0) for r in failed.values()),
                        "completion_tokens": sum(int(r.n_completion_tokens or 0) for r in failed.values()),
                        "wall_s": 0.0,  # the chunk's wall time is already on the main row
                    }
                )
            self.store.write_chunk(gen_rows, cells, ledgers, shard_writer=self.shard_writer, stage=stage)
            self.chunks_committed += 1
            self.chunks_executed += int(bool(chunk))
            self.n_requested += n_req
            self.n_executed += len(gen_rows)
            done += n_req
            if failed:
                sample = next(iter(failed.values())).text[:200]
                raise GenerationError(
                    f"{len(failed)} generation(s) failed in chunk {ci} of stage {stage!r} (not cached; "
                    f"re-run to retry). First error: {sample}"
                )
            if self.on_progress is not None:
                self.on_progress(
                    {
                        "stage": stage,
                        "purpose": purpose,
                        "seed": seed,
                        "round": round_,
                        "chunk": ci + 1,
                        "n_chunks": len(chunks),
                        "n_tasks": len(tasks),
                        "n_done": done,
                        "n_executed": len(gen_rows),
                        "n_cache_hits": ledger["n_cache_hits"],
                        "wall_s": ledger["wall_s"],
                    }
                )
        return records  # type: ignore[return-value]
