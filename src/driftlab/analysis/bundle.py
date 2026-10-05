"""``driftlab analyze``: every post-hoc analysis of one run dir, persisted as an :class:`AnalysisBundle`.

:func:`analyze` reads ``<run_dir>/config.yaml``, the analysis plan (``<run_dir>/plan.yaml``, ``plan_path`` or
the config's ``analysis_plan``) and ``<run_dir>/store.sqlite`` (read-only), then builds every frame of
:data:`driftlab.analysis.bundle_io.FRAME_NAMES` (no model calls) and saves them under
``<run_dir>/exports/analysis/``. Optionally a small JSON digest is written to the store's ``analysis_results``
table (a separate read-write connection; failures there only add a warning).

Small or partial runs never crash the bundle: missing cells become skipped pairs / skipped (seed, policy)
simulations, ages beyond R give empty T7 rows, and a derived frame whose producer fails is emitted empty with
the error recorded in ``meta["warnings"]`` and ``meta["empty_frames"]``. Frames that are legitimately empty:

* ``audit``: no ``audit_results`` rows (audit disabled or the audit stage has not run);
* ``pairs`` / ``pair_summary``: no completed trajectory round with matching eval cells (``table4``/``table7``
  keep their rows with ``n_pairs = 0``);
* ``policy_*`` / ``candidates`` / ``ablations`` / ``schedule_random``: no seed with a complete eval matrix for
  its trajectory, or (``ablations``) a plan without ablations, or (``schedule_random``) fewer completed rounds
  than ``plan.schedule_randomization.n_changes``;
* ``ledger_summary``: no ledger rows (e.g. a hand-built store); ``truncation`` / ``accuracy``: no eval cells.
"""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import driftlab
from driftlab.analysis import loader
from driftlab.analysis.ablations import (
    ABLATION_COLUMNS,
    SCHEDULE_RANDOM_COLUMNS,
    run_ablations,
    schedule_randomization_frame,
)
from driftlab.analysis.allpairs import (
    PAIR_COLUMNS,
    T4_COLUMNS,
    T7_COLUMNS,
    all_pairs,
    summarize_pairs,
    table4_frame,
    table7_frame,
)
from driftlab.analysis.allpairs import (
    SUMMARY_COLUMNS as PAIR_SUMMARY_COLUMNS,
)
from driftlab.analysis.bundle_io import FRAME_NAMES, SYNTHETIC_NOTE, AnalysisBundle, save_bundle
from driftlab.analysis.cube import Cube, MissingCell, Trajectory
from driftlab.analysis.hypotheses import HYPOTHESIS_COLUMNS, evaluate_hypotheses
from driftlab.analysis.policies import (
    CANDIDATE_COLUMNS,
    REF_COLUMNS,
    ROUND_COLUMNS,
    PolicyRun,
    candidates_frame,
    decision_policies,
    parse_policy,
    runs_to_frames,
    simulate_all,
    summarize_policies,
)
from driftlab.analysis.policies import (
    SUMMARY_COLUMNS as POLICY_SUMMARY_COLUMNS,
)
from driftlab.config import AnalysisPlan, ExperimentConfig, load_config, load_plan
from driftlab.environments import Environment, diff
from driftlab.extraction import extractor_tags as current_extractor_tags
from driftlab.store.store import Store, utc_now

CONFIG_FILE = "config.yaml"
PLAN_FILE = "plan.yaml"
STORE_FILE = "store.sqlite"
PROVENANCE_FILE = "provenance.json"
ANALYSIS_RESULT_NAME = "bundle_summary"

LEDGER_COLUMNS: tuple[str, ...] = (
    "stage",
    "purpose",
    "seed",
    "n_requested",
    "n_executed",
    "n_cache_hits",
    "prompt_tokens",
    "completion_tokens",
    "wall_s",
    "n_rows",
)
AUDIT_COLUMNS: tuple[str, ...] = (
    "decoding_id",
    "slot",
    "repeat",
    "pct_text_identical",
    "pct_correct_flip",
    "n",
)
TRUNCATION_COLUMNS: tuple[str, ...] = (
    "seed",
    "decoding",
    "slot",
    "draw_kind",
    "draw",
    "trunc_rate",
    "n_items",
    "n_truncated",
    "physical",
)
TRAJECTORY_COLUMNS: tuple[str, ...] = (
    "seed",
    "round",
    "incumbent_slot",
    "candidate_slot",
    "inc_dev_acc",
    "cand_dev_acc",
    "advanced",
    "n_attempts",
    "is_fallback",
    "origin",
    "prompt_hash",
    "prompt_text",
    "incumbent_after",
)
ENVIRONMENT_COLUMNS: tuple[str, ...] = (
    "env_id",
    "decoding",
    "temperature",
    "extractor",
    "extractor_tag",
    "fingerprint",
    "description",
    "changed_vs_storage",
    "top_p",
    "top_k",
    "repetition_penalty",
    "max_new_tokens",
    "is_storage_env",
)
ACCURACY_COLUMNS: tuple[str, ...] = (
    "seed",
    "env",
    "slot",
    "round",
    "gt_acc",
    "decoding",
    "extractor",
    "is_incumbent",
)

FRAME_COLUMNS: dict[str, tuple[str, ...]] = {
    "pairs": PAIR_COLUMNS,
    "pair_summary": ("env", "age", *PAIR_SUMMARY_COLUMNS),
    "table4": T4_COLUMNS,
    "table7": T7_COLUMNS,
    "policy_rounds": ROUND_COLUMNS,
    "policy_refs": REF_COLUMNS,
    "policy_summary": POLICY_SUMMARY_COLUMNS,
    "candidates": CANDIDATE_COLUMNS,
    "ablations": ABLATION_COLUMNS,
    "schedule_random": SCHEDULE_RANDOM_COLUMNS,
    "hypotheses": HYPOTHESIS_COLUMNS,
    "ledger_summary": LEDGER_COLUMNS,
    "audit": AUDIT_COLUMNS,
    "truncation": TRUNCATION_COLUMNS,
    "trajectory": TRAJECTORY_COLUMNS,
    "environments": ENVIRONMENT_COLUMNS,
    "accuracy": ACCURACY_COLUMNS,
}
EMPTY_REASONS: dict[str, str] = {
    "pairs": "no completed trajectory round with matching eval cells",
    "pair_summary": "no pairs",
    "policy_rounds": "no seed with a complete eval matrix for its trajectory",
    "policy_refs": "no policy runs",
    "policy_summary": "no policy runs",
    "candidates": "no policy runs",
    "ablations": "no ablations in the plan, or no simulable seed",
    "schedule_random": (
        "no simulable seed, schedule_randomization.n = 0, or fewer completed rounds than "
        "schedule_randomization.n_changes"
    ),
    "hypotheses": "hypotheses not evaluated",
    "ledger_summary": "no ledger rows",
    "audit": "no audit_results rows (audit disabled or not run)",
    "truncation": "no eval cells",
    "trajectory": "no trajectory (no slot 0)",
    "environments": "no environments",
    "accuracy": "no eval cells",
    "table4": "no environments",
    "table7": "no T7 ages",
}

Log = Callable[[str], None]


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=float) for c in columns})


def _finite(v: object) -> object:
    """JSON-friendly scalar: NaN/inf -> None, numpy scalars -> Python."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


# --------------------------------------------------------------------------- inputs


def _plan_file(run_dir: Path, cfg: ExperimentConfig, plan_path: str | Path | None) -> Path:
    if plan_path is not None:
        return Path(plan_path)
    local = run_dir / PLAN_FILE
    return local if local.exists() else cfg.resolve_path(cfg.analysis_plan)


def _run_row(store: Store, cfg: ExperimentConfig) -> dict:
    row = store.get_run(cfg.run.name)
    if row is None:
        row = store.get_run(None)  # the database's only run (LookupError if several)
    if row is None:
        raise LookupError(f"{store.path}: no run in the database")
    return row


def _provenance(run_dir: Path, run: Mapping[str, Any]) -> dict:
    import json

    path = run_dir / PROVENANCE_FILE
    for raw in ((path.read_text(encoding="utf-8") if path.exists() else None), run.get("provenance_json")):
        if not raw:
            continue
        try:
            out = json.loads(raw)
        except ValueError:
            continue
        if isinstance(out, dict):
            return out
    return {}


def policy_names(plan: AnalysisPlan) -> list[str]:
    """Policies simulated under the headline schedule: ``plan.policies`` + headline policies + the FIXEDAGE
    policies the ablations use (canonical names, first-appearance order)."""
    names = [parse_policy(p).name for p in (*plan.policies, *plan.headline_policies)]
    for abl in plan.ablations.values():
        if abl.fixed_age is not None:
            names.append(parse_policy(f"FIXEDAGE_k{int(abl.fixed_age)}").name)
    return list(dict.fromkeys(names))


# --------------------------------------------------------------------------- simple frames


def ledger_summary_frame(ledger: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Ledger rows aggregated by (stage, purpose, seed); a NULL seed (all seeds) is reported as -1."""
    if not ledger:
        return _empty(LEDGER_COLUMNS)
    df = pd.DataFrame(list(ledger))
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").fillna(-1).astype(np.int64)
    num = ["n_requested", "n_executed", "n_cache_hits", "prompt_tokens", "completion_tokens", "wall_s"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["stage"] = df["stage"].astype(str)
    df["purpose"] = df["purpose"].astype(str)
    g = df.groupby(["stage", "purpose", "seed"], sort=True)
    out = g[num].sum().reset_index()
    out["n_rows"] = g.size().to_numpy()
    for c in num[:-1]:
        out[c] = out[c].astype(np.int64)
    return out[list(LEDGER_COLUMNS)]


def audit_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if not rows:
        return _empty(AUDIT_COLUMNS)
    df = pd.DataFrame(list(rows))
    return df[[*AUDIT_COLUMNS, *(c for c in df.columns if c not in AUDIT_COLUMNS and c != "run_id")]]


def truncation_frame(cube: Cube) -> pd.DataFrame:
    """Truncation rate (``finish_reason == "length"``) of every cell present in the cube."""
    present = cube.gen_row >= 0
    n_present = present.sum(axis=-1)
    n_trunc = (cube.truncated & present).sum(axis=-1)
    s, d, k, r = np.nonzero(n_present > 0)
    if not len(s):
        return _empty(TRUNCATION_COLUMNS)
    kinds = np.array([dr[0] for dr in cube.draws], dtype=object)
    draws = np.array([dr[1] for dr in cube.draws], dtype=np.int64)
    n = n_present[s, d, k, r]
    t = n_trunc[s, d, k, r]
    return pd.DataFrame(
        {
            "seed": np.asarray(cube.seeds, dtype=np.int64)[s],
            "decoding": np.asarray(cube.decodings, dtype=object)[d],
            "slot": k.astype(np.int64),
            "draw_kind": kinds[r],
            "draw": draws[r],
            "trunc_rate": t / n,
            "n_items": n.astype(np.int64),
            "n_truncated": t.astype(np.int64),
            "physical": cube.physical[s, d, k, r],
        },
        columns=list(TRUNCATION_COLUMNS),
    )


def trajectory_frame(
    trajs: Mapping[int, Trajectory], attempts: Mapping[int, Mapping[int, int]]
) -> pd.DataFrame:
    """One row per (seed, round) incl. round 0 (round 0: slot 0 as both incumbent and candidate, 0 attempts)."""
    rows = []
    for seed in sorted(trajs):
        tr = trajs[seed]
        for t in range(tr.R + 1):
            rows.append(
                {
                    "seed": int(seed),
                    "round": t,
                    "incumbent_slot": int(tr.inc_slot[t]),
                    "candidate_slot": t,
                    "inc_dev_acc": float(tr.inc_dev_acc[t]),
                    "cand_dev_acc": float(tr.cand_dev_acc[t]),
                    "advanced": bool(tr.advanced[t]),
                    "n_attempts": int(attempts.get(seed, {}).get(t, 0)) if t else 0,
                    "is_fallback": bool(tr.is_fallback[t]),
                    "origin": tr.origin[t] if t < len(tr.origin) else "",
                    "prompt_hash": tr.prompt_hashes[t],
                    "prompt_text": tr.prompts[t],
                    "incumbent_after": int(tr.incumbent_after(t)),
                }
            )
    return pd.DataFrame(rows, columns=list(TRAJECTORY_COLUMNS)) if rows else _empty(TRAJECTORY_COLUMNS)


def environments_frame(
    envs: Mapping[str, Environment], storage_env: str, tags: Mapping[str, str]
) -> pd.DataFrame:
    """Table 2 source: one row per environment; ``changed_vs_storage`` lists the components that differ."""
    st = envs.get(storage_env)
    rows = []
    for eid, env in envs.items():
        d = env.decoding
        rows.append(
            {
                "env_id": eid,
                "decoding": d.id,
                "temperature": float(d.temperature),
                "extractor": env.extractor,
                "extractor_tag": tags.get(env.extractor, env.extractor),
                "fingerprint": env.fingerprint(tags),
                "description": env.description,
                "changed_vs_storage": ",".join(diff(st, env, tags)) if st is not None else "",
                "top_p": float(d.top_p),
                "top_k": int(d.top_k),
                "repetition_penalty": float(d.repetition_penalty),
                "max_new_tokens": int(d.max_new_tokens),
                "is_storage_env": eid == storage_env,
            }
        )
    return pd.DataFrame(rows, columns=list(ENVIRONMENT_COLUMNS)) if rows else _empty(ENVIRONMENT_COLUMNS)


def accuracy_frame(
    cube: Cube, trajs: Mapping[int, Trajectory], envs: Mapping[str, Environment], plan: AnalysisPlan
) -> pd.DataFrame:
    """GT accuracy (plan GT mode; ``split_half`` -> second item half) of every slot k at rounds r >= k."""
    N = cube.n_items
    mode = plan.gt.mode
    gt_items = np.arange(N // 2, N) if mode == "split_half" else None
    gt_mode = "independent_draw" if mode == "split_half" else mode
    rows = []
    for seed in cube.seeds:
        tr = trajs.get(int(seed))
        for eid, env in envs.items():
            dec, x = env.decoding.id, env.extractor
            for k in range(cube.n_slots):
                for r in range(k, cube.n_slots):
                    try:
                        acc = cube.gt_acc(int(seed), dec, k, r, x, mode=gt_mode, items=gt_items)
                    except (MissingCell, KeyError, IndexError):
                        continue
                    inc = tr is not None and r <= tr.R and int(tr.inc_slot[r]) == k
                    rows.append(
                        {
                            "seed": int(seed),
                            "env": eid,
                            "slot": k,
                            "round": r,
                            "gt_acc": acc,
                            "decoding": dec,
                            "extractor": x,
                            "is_incumbent": bool(inc),
                        }
                    )
    return pd.DataFrame(rows, columns=list(ACCURACY_COLUMNS)) if rows else _empty(ACCURACY_COLUMNS)


# --------------------------------------------------------------------------- analyze


class _Collector:
    """Warnings, empty-frame reasons and guarded producers of one analyze() call."""

    def __init__(self, log: Log) -> None:
        self.log = log
        self.warnings: list[str] = []
        self.failed: dict[str, str] = {}

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        self.log(f"[analyze] WARNING: {msg}")

    def run(self, name: str, fn: Callable[[], Any], default: Any) -> Any:
        """Run a producer; driftlab UserWarnings are recorded, an exception yields ``default`` + a warning."""
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                out = fn()
        except Exception as e:  # a derived product must not abort the bundle (partial runs)
            msg = f"{name} failed: {type(e).__name__}: {e}"
            self.failed[name] = msg
            self.warn(msg)
            return default
        for w in caught:
            if issubclass(w.category, UserWarning) and "driftlab" in str(w.filename):
                self.warn(f"{name}: {w.message}")
        return out


def _coverage_warnings(
    col: _Collector, cube: Cube, trajs: Mapping[int, Trajectory], envs, db: Path, run_id: str
):
    no_cells = sorted(set(trajs) - {int(s) for s in cube.seeds})
    if no_cells:
        col.warn(f"seeds {no_cells} have a trajectory but no eval cells")
    no_traj = sorted({int(s) for s in cube.seeds} - set(trajs))
    if no_traj:
        col.warn(f"seeds {no_traj} have eval cells but no trajectory")
    present = cube.gen_row >= 0
    n_present = int(present.sum())
    for x in sorted({e.extractor for e in envs.values()}):
        if not n_present:
            break
        if x not in cube.extractors:
            col.warn(
                f"extractor {x}: no score rows for any of the {n_present} eval cell items (run the score stage)"
            )
            continue
        unscored = int((present & (cube.correct[:, :, :, :, cube.extractors.index(x), :] < 0)).sum())
        if unscored:
            col.warn(
                f"extractor {x}: {unscored} of {n_present} eval cell items have a generation but no score "
                "(run the score stage)"
            )
    stale = loader.score_status(db, run_id)
    stale = stale.loc[stale["stale"].astype(bool)] if len(stale) else stale
    for r in stale.itertuples(index=False):
        col.warn(
            f"{int(r.n)} score rows of extractor {r.extractor} use source hash {r.ext_hash}, current is "
            f"{r.current_hash} (re-run the score stage)"
        )


def analyze(
    run_dir: str | Path,
    *,
    plan_path: str | Path | None = None,
    B: int | None = None,
    store_results: bool = True,
    log: Log = print,
    schedule_random_n: int | None = None,
    B_policy: int | None = None,
) -> AnalysisBundle:
    """Build, save and return the analysis bundle of ``run_dir`` (see the module docstring).

    ``B`` overrides ``plan.bootstrap.B`` (every bootstrap). The H2/H3 re-simulation bootstrap uses the same
    pre-registered ``B`` unless ``B_policy`` caps it (``min(B_policy, B)`` replicates; e.g. for quick looks);
    ``schedule_random_n`` overrides ``plan.schedule_randomization.n`` (exploratory frame; smaller is faster).
    """
    t0 = time.perf_counter()
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / CONFIG_FILE)
    plan_file = _plan_file(run_dir, cfg, plan_path)
    plan = load_plan(plan_file)
    B = int(plan.bootstrap.B if B is None else B)
    bseed = int(plan.bootstrap.seed)
    b_policy = max(0, B if B_policy is None else min(int(B_policy), B))
    db = run_dir / STORE_FILE
    if not db.exists():
        raise FileNotFoundError(f"no store at {db}")
    col = _Collector(log)

    with Store(db, read_only=True) as store:
        run = _run_row(store, cfg)
        run_id = str(run["run_id"])
        ledger = store.ledger_rows(run_id)
        audit_rows = store.get_audit_results(run_id)
    cube = loader.load_cube(db, run_id)
    trajs = loader.load_trajectories(db, run_id)
    attempts = loader.proposer_attempts(db, run_id)
    log(
        f"[analyze] run {run_id}: cube {len(cube.seeds)} seeds x {len(cube.decodings)} decodings x "
        f"{cube.n_slots} slots x {len(cube.draws)} draws x {len(cube.extractors)} extractors x {cube.n_items} items; "
        f"trajectories {sorted(trajs)} ({time.perf_counter() - t0:.1f}s)"
    )
    synthetic = bool(run["synthetic"]) or cube.synthetic or cfg.backend.kind == "mock"
    envs = cfg.all_environments(engine_fp=str(run.get("engine_fp") or ""))
    tags = current_extractor_tags()
    n_dev = int(cfg.data.dev.n)
    _coverage_warnings(col, cube, trajs, envs, db, run_id)
    if run.get("config_hash") and run["config_hash"] != cfg.config_hash():
        col.warn(f"config.yaml hashes to {cfg.config_hash()}, the run was recorded with {run['config_hash']}")
    if run.get("plan_hash") and run["plan_hash"] != plan.plan_hash():
        col.warn(
            f"analysis plan {plan.plan_hash()} differs from the plan recorded at run time ({run['plan_hash']})"
        )
    observed = {s: t.R for s, t in trajs.items()}
    if any(r < cfg.run.rounds for r in observed.values()):
        col.warn(f"partial trajectories: completed rounds per seed {observed} < R={cfg.run.rounds}")
    if cube.n_items != int(cfg.data.eval.n):
        col.warn(
            f"the store holds {cube.n_items} eval items but the config asks for {cfg.data.eval.n}; items without "
            "cells count as missing"
        )

    frames: dict[str, pd.DataFrame] = {}
    empty_reasons: dict[str, str] = {}  # data-specific reasons that override EMPTY_REASONS
    pr = all_pairs(cube, trajs, envs, plan)
    if pr.skipped:
        col.warn(f"{len(pr.skipped)} pairs skipped for missing cells (first: {pr.skipped[0]})")
    frames["pairs"] = pr.df
    summary = col.run("pair_summary", lambda: summarize_pairs(pr, by=("env", "age"), B=B, seed=bseed), None)
    frames["pair_summary"] = summary if summary is not None else _empty(FRAME_COLUMNS["pair_summary"])
    frames["table4"] = table4_frame(frames["pair_summary"], plan, env_ids=list(envs))
    frames["table7"] = table7_frame(frames["pair_summary"], plan)
    log(
        f"[analyze] all-pairs: {len(pr.df)} pairs, {len(pr.skipped)} skipped ({time.perf_counter() - t0:.1f}s)"
    )

    names = policy_names(plan)
    sim_trajs = {s: t for s, t in trajs.items() if s in set(cube.seeds) and cube.n_slots > t.R}
    for s in sorted(set(trajs) - set(sim_trajs)):
        if s in set(cube.seeds):
            col.warn(
                f"seed {s}: trajectory has {trajs[s].R} rounds but the eval matrix covers {cube.n_slots} slots"
            )
    runs: list[PolicyRun] = col.run(
        "policies",
        lambda: simulate_all(
            cube, sim_trajs, plan, envs, n_dev, tags, attempts, policies=names, on_missing="skip"
        ),
        [],
    )
    have = {(r.policy.name, int(r.seed)) for r in runs}
    ok_trajs = {s: t for s, t in sim_trajs.items() if all((n, int(s)) in have for n in names)}
    rounds_df, refs_df = runs_to_frames(runs)
    frames["policy_rounds"], frames["policy_refs"] = rounds_df, refs_df
    frames["policy_summary"] = summarize_policies(runs)
    dec_cols = (*CANDIDATE_COLUMNS, *decision_policies(plan))
    frames["candidates"] = col.run(
        "candidates",
        lambda: candidates_frame(cube, ok_trajs, plan, envs, runs, extractor_tags=tags),
        _empty(dec_cols),
    )
    log(
        f"[analyze] policies: {len(runs)} runs over seeds {sorted(ok_trajs)} ({time.perf_counter() - t0:.1f}s)"
    )
    frames["ablations"] = col.run(
        "ablations",
        lambda: (
            run_ablations(
                cube,
                ok_trajs,
                plan,
                envs,
                pr.df,
                n_dev,
                tags,
                proposer_attempts_by_seed=attempts,
                on_missing="skip",
            )
            if ok_trajs
            else _empty(ABLATION_COLUMNS)
        ),
        _empty(ABLATION_COLUMNS),
    )
    n_changes = int(plan.schedule_randomization.n_changes)
    r_min = min((t.R for t in ok_trajs.values()), default=0)
    if ok_trajs and r_min < n_changes:  # randomize_schedules needs n_changes distinct change rounds in 1..R
        empty_reasons["schedule_random"] = (
            f"fewer completed rounds (R={r_min}) than schedule_randomization.n_changes={n_changes}"
        )
        frames["schedule_random"] = _empty(SCHEDULE_RANDOM_COLUMNS)
    else:
        frames["schedule_random"] = col.run(
            "schedule_random",
            lambda: (
                schedule_randomization_frame(
                    cube,
                    ok_trajs,
                    plan,
                    envs,
                    n_dev,
                    extractor_tags=tags,
                    n=schedule_random_n,
                    proposer_attempts_by_seed=attempts,
                    on_missing="skip",
                )
                if ok_trajs
                else _empty(SCHEDULE_RANDOM_COLUMNS)
            ),
            _empty(SCHEDULE_RANDOM_COLUMNS),
        )
    log(f"[analyze] ablations + schedule randomization ({time.perf_counter() - t0:.1f}s)")
    frames["hypotheses"] = col.run(
        "hypotheses",
        lambda: evaluate_hypotheses(
            pr,
            runs,
            cube,
            trajs,
            plan,
            envs,
            n_dev=n_dev,
            extractor_tags=tags,
            B=B,
            seed=bseed,
            B_policy=b_policy,
        ),
        _empty(HYPOTHESIS_COLUMNS),
    )
    log(f"[analyze] hypotheses ({time.perf_counter() - t0:.1f}s)")
    frames["ledger_summary"] = ledger_summary_frame(ledger)
    frames["audit"] = audit_frame(audit_rows)
    frames["truncation"] = truncation_frame(cube)
    frames["trajectory"] = trajectory_frame(trajs, attempts)
    frames["environments"] = environments_frame(envs, str(plan.storage_env), tags)
    frames["accuracy"] = col.run(
        "accuracy", lambda: accuracy_frame(cube, trajs, envs, plan), _empty(ACCURACY_COLUMNS)
    )
    frames = {name: frames.get(name, _empty(FRAME_COLUMNS.get(name, ()))) for name in FRAME_NAMES}

    empty = {
        n: col.failed.get(n, empty_reasons.get(n, EMPTY_REASONS.get(n, "no data")))
        for n, df in frames.items()
        if df is None or df.empty
    }
    meta: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "config_hash": str(run.get("config_hash") or cfg.config_hash()),
        "plan_hash": plan.plan_hash(),
        "plan_version": plan.version,
        "synthetic": synthetic,
        "synthetic_note": SYNTHETIC_NOTE if synthetic else "",
        "created_at": utc_now(),
        "seeds": [int(s) for s in cube.seeds],
        "R": int(cfg.run.rounds),
        "R_observed": {str(s): int(t.R) for s, t in sorted(trajs.items())},
        "N": int(cube.n_items),
        "D": n_dev,
        "schedule": {str(k): str(v) for k, v in sorted(plan.schedule.items())},
        "extractor_tags": dict(tags),
        "engine_fp": run.get("engine_fp"),
        "config": cfg.model_dump(mode="json"),
        "plan": plan.model_dump(mode="json"),
        "plan_file": str(plan_file),
        "provenance": _provenance(run_dir, run),
        "driftlab_version": driftlab.__version__,
        "warnings": list(col.warnings),
        "skipped_pairs": int(len(pr.skipped)),
        "policies": names,
        "B": B,
        "B_policy": b_policy,
        "bootstrap_seed": bseed,
        "gt_mode": plan.gt.mode,
        "cube_shape": [int(v) for v in cube.correct.shape],
        "empty_frames": empty,
        "frame_rows": {n: int(len(df)) for n, df in frames.items()},
    }
    bundle = AnalysisBundle(meta=meta, frames=frames)
    if store_results:
        _store_summary(db, bundle, col)
    meta["warnings"] = list(col.warnings)
    meta["elapsed_s"] = round(time.perf_counter() - t0, 3)
    out = save_bundle(bundle, run_dir)
    log(f"[analyze] bundle written to {out} ({meta['elapsed_s']:.1f}s, {len(col.warnings)} warnings)")
    return bundle


def _store_summary(db: Path, bundle: AnalysisBundle, col: _Collector) -> None:
    """Write a small JSON digest to ``analysis_results`` (key = plan hash); failures only warn."""
    hyp = bundle.frame("hypotheses")
    payload = {
        "run_id": bundle.meta["run_id"],
        "created_at": bundle.meta["created_at"],
        "synthetic": bundle.meta["synthetic"],
        "config_hash": bundle.meta["config_hash"],
        "frame_rows": bundle.meta["frame_rows"],
        "n_warnings": len(col.warnings),
        "hypotheses": [
            {k: _finite(v) for k, v in rec.items()}
            for rec in hyp[["hypothesis", "subject", "estimate", "ci_lo", "ci_hi", "unit", "status"]].to_dict(
                "records"
            )
        ]
        if len(hyp)
        else [],
    }
    try:
        with Store(db) as rw:
            rw.put_analysis_result(bundle.plan_hash, ANALYSIS_RESULT_NAME, payload)
    except Exception as e:  # read-only media, locked DB, ...: the bundle on disk is the product
        col.warn(f"analysis_results not written: {type(e).__name__}: {e}")


# --------------------------------------------------------------------------- CLI digest


def _num(v: object) -> float:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return f


def _int(v: object) -> int:
    """An int for display; NaN / missing / unparsable -> 0 (``int(nan or 0)`` would raise: NaN is truthy)."""
    f = _num(v)
    return int(f) if math.isfinite(f) else 0


def _fmt(v: object, digits: int = 1) -> str:
    f = _num(v)
    return "—" if not math.isfinite(f) else f"{f:+.{digits}f}"


def _ci(lo: object, hi: object, digits: int = 1) -> str:
    a, b = _num(lo), _num(hi)
    return "" if not (math.isfinite(a) and math.isfinite(b)) else f" [{a:.{digits}f}, {b:.{digits}f}]"


def _truthy(v: object) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return False if v is None or (isinstance(v, float) and math.isnan(v)) else bool(v)


def main_summary(bundle: AnalysisBundle) -> str:
    """Short plain-text digest of H1-H3, T3 (headline policies) and T7 for CLI output."""
    m = bundle.meta
    lines = [
        f"DriftLab analysis: run {m.get('run_id', '?')}"
        + (f"  [{SYNTHETIC_NOTE}]" if bundle.synthetic else ""),
        f"plan {m.get('plan_version', '?')} ({m.get('plan_hash', '?')}), config {m.get('config_hash', '?')}; "
        f"seeds {m.get('seeds', [])}, R={m.get('R', '?')}, N={m.get('N', '?')}",
    ]
    hyp = bundle.frame("hypotheses")
    if len(hyp):
        for h in ("H1", "H2", "H3"):
            sub = hyp.loc[hyp["hypothesis"].astype(str) == h]
            if not len(sub):
                continue
            overall = sub.loc[sub["subject"].astype(str) == "overall"]
            head = f"{h}: {overall.iloc[0]['status']}" if len(overall) else f"{h}:"
            parts = []
            for r in sub.itertuples(index=False):
                if str(r.subject) == "overall" or str(r.status) == "descriptive":
                    continue
                dag = " ‡" if "‡" in str(getattr(r, "note", "")) else ""  # (partly) zero by construction
                if str(r.unit) == "ratio":
                    est = "—" if not math.isfinite(_num(r.estimate)) else f"{_num(r.estimate):.2f}"
                    parts.append(f"{r.contrast} = {est} ({r.status}){dag}")
                    continue
                parts.append(f"{r.contrast} = {_fmt(r.estimate)} pp{_ci(r.ci_lo, r.ci_hi)} ({r.status}){dag}")
            lines.append(head + ("\n    " + "\n    ".join(parts) if parts else ""))
        if "note" in hyp.columns and hyp["note"].astype(str).str.contains("‡", regex=False).any():
            lines.append(
                "    (‡ = some compared values are identical / zero by construction; see hypotheses notes)"
            )
    else:
        lines.append("hypotheses: not evaluated")
    pol = bundle.frame("policy_summary")
    plan = m.get("plan") or {}
    headline = [str(p) for p in plan.get("headline_policies", ["P1", "P2", "P3"])]
    if len(pol):
        lines.append("T3 (headline policies; FAR‡ = every accept GT-coupled, i.e. 0 by construction):")
        for p in headline:
            sub = pol.loc[pol["policy"].astype(str) == p]
            if not len(sub):
                continue
            r = sub.iloc[0]
            far = _num(r.get("far_pooled"))
            n_acc = _int(r.get("n_accepted"))
            coupled = n_acc > 0 and _int(r.get("n_accepted_gt_coupled")) == n_acc  # FAR 0 by construction
            far_s = "—" if not math.isfinite(far) else f"{far:.2f}" + ("‡" if coupled else "")
            lines.append(
                f"    {p:<6} GT acc {_num(r.get('gt_acc_mean')):.3f} ± {_num(r.get('gt_acc_sd')):.3f}  "
                f"FAR {far_s} ({_int(r.get('n_false_accepts'))}/{n_acc})  "
                f"calls {_num(r.get('calls_total')):g} (reference {_num(r.get('calls_reference')):g})"
            )
    t7 = bundle.frame("table7")
    if len(t7):
        lines.append("T7 (inflation pp by reference age; ‡ = by construction):")
        for age, g in t7.groupby("age", sort=True):
            cells = []
            for r in g.itertuples(index=False):
                if _int(r.n_pairs) == 0:
                    cells.append(f"{r.env} —")
                    continue
                dag = "‡" if _truthy(r.by_construction) else ""
                cells.append(
                    f"{r.env} {_num(r.inflation) * 100:+.1f}{dag}{_ci(_num(r.infl_lo) * 100, _num(r.infl_hi) * 100)}"
                )
            lines.append(f"    age {int(age):>2}: " + "   ".join(cells))
    warns = list(m.get("warnings") or [])
    if warns:
        lines.append(f"warnings ({len(warns)}):")
        lines.extend(f"    - {w}" for w in warns[:5])
        if len(warns) > 5:
            lines.append(f"    ... {len(warns) - 5} more in bundle.json")
    return "\n".join(lines)


__all__ = [
    "ACCURACY_COLUMNS",
    "AUDIT_COLUMNS",
    "EMPTY_REASONS",
    "ENVIRONMENT_COLUMNS",
    "FRAME_COLUMNS",
    "LEDGER_COLUMNS",
    "TRAJECTORY_COLUMNS",
    "TRUNCATION_COLUMNS",
    "accuracy_frame",
    "analyze",
    "audit_frame",
    "environments_frame",
    "ledger_summary_frame",
    "main_summary",
    "policy_names",
    "trajectory_frame",
    "truncation_frame",
]
