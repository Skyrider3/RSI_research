"""Ablations A1-A4 (T8) and schedule randomization (docs/ARCHITECTURE.md section 6.4).

* :func:`run_ablations` re-runs the policy simulation under each ablation schedule of the plan (A4 replaces
  P3 by ``FIXEDAGE_k<fixed_age>``) and reads drift inflation from the factorial all-pairs frame over the
  environments the schedule visits after its first change.
* :func:`randomize_schedules` / :func:`schedule_randomization_frame` are **EXPLORATORY** (not part of the
  pre-registered contrasts): random schedules with exactly ``n_changes`` change rounds, and the distribution of
  FAR and calls per policy over them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from driftlab.analysis.cube import Cube, Trajectory
from driftlab.analysis.policies import (
    PolicyRun,
    as_environments,
    as_trajectories,
    parse_policy,
    simulate_all,
)
from driftlab.environments import Environment, EnvSchedule
from driftlab.keys import rng_seed

ABLATION_COLUMNS: tuple[str, ...] = (
    "ablation",
    "description",
    "schedule",
    "decoding_changes",
    "extraction_changes",
    "reference_age_variation",
    "inflation_age",
    "inflation_envs",
    "n_pairs",
    "drift_inflation_pp",
    "far_P1",
    "far_P1b",
    "far_P3",
    "p3_policy",
    "far_P1_seed_mean",
    "far_P1b_seed_mean",
    "far_P3_seed_mean",
    "n_accepted_P1",
    "n_accepted_P1b",
    "n_accepted_P3",
    "gt_acc_P3",
    "calls_P3",
)

SCHEDULE_RANDOM_COLUMNS: tuple[str, ...] = (
    "schedule_id",
    "schedule",
    "change_rounds",
    "seed",
    "policy",
    "far",
    "n_accepted",
    "n_false_accepts",
    "calls_total",
    "reference_calls",
    "refreshes",
    "rescores",
    "gt_acc",
)


def format_schedule(changes: Mapping[int, str] | EnvSchedule) -> str:
    """``{0: "E1", 4: "E2"}`` -> ``"0:E1, 4:E2"`` (rounds ascending)."""
    ch = changes.changes if isinstance(changes, EnvSchedule) else changes
    return ", ".join(f"{int(r)}:{ch[r]}" for r in sorted(ch, key=int))


def _rounds(trajs: Mapping[int, Trajectory], cube: Cube) -> int:
    return max((t.R for t in trajs.values()), default=cube.R)


def schedule_flags(schedule: EnvSchedule, envs: Mapping[str, Environment], R: int) -> dict[str, object]:
    """Which components the schedule changes within rounds 0..R.

    ``decoding_changes`` / ``extraction_changes``: some visited env's decoding parameters / extractor differ
    from round 0's; ``first_change``: first change round (None without changes); ``envs_after_change``:
    env ids visited at rounds >= the first change (sorted).
    """
    ids = schedule.as_list(R)
    e0 = envs[ids[0]]
    visited = [envs[e] for e in ids]
    first = next((t for t in range(1, R + 1) if ids[t] != ids[t - 1]), None)
    after = sorted(set(ids[first:])) if first is not None else []
    return {
        "decoding_changes": any(e.decoding.params() != e0.decoding.params() for e in visited),
        "extraction_changes": any(e.extractor != e0.extractor for e in visited),
        "first_change": first,
        "envs_after_change": after,
    }


def _far_stats(runs: Sequence[PolicyRun]) -> tuple[float, float, int]:
    """(pooled FAR, per-seed mean FAR (NaN-aware), n accepted)."""
    n_acc = sum(r.n_accepted for r in runs)
    n_fa = sum(r.n_false_accepts for r in runs)
    per_seed = [r.far for r in runs if np.isfinite(r.far)]
    pooled = n_fa / n_acc if n_acc else float("nan")
    return pooled, (float(np.mean(per_seed)) if per_seed else float("nan")), int(n_acc)


def _mean_calls(runs: Sequence[PolicyRun]) -> float | int:
    vals = [r.total_calls for r in runs]
    if not vals:
        return float("nan")
    return int(vals[0]) if all(v == vals[0] for v in vals) else float(np.mean(vals))


def run_ablations(
    cube: Cube,
    trajs: Mapping[int, Trajectory] | Sequence[Trajectory],
    plan,
    envs: Mapping[str, Environment] | None,
    pairs_df: pd.DataFrame,
    n_dev: int,
    extractor_tags: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """T8: one row per ablation of ``plan.ablations`` (plan order); columns :data:`ABLATION_COLUMNS`.

    * ``drift_inflation_pp``: mean of ``pairs_df.inflation * 100`` over pairs with ``age == fixed_age`` (if
      set) else ``plan.t4.age`` whose ``env`` is visited by the schedule at or after its first change
      (``inflation_envs``); NaN when the schedule has no change or no pair matches. Only the ``env``, ``age``
      and ``inflation`` columns of ``pairs_df`` are used (one row per seed, env, j, i).
    * ``far_*``: pooled FAR (sum false accepts / sum accepts over seeds); ``*_seed_mean``: per-seed mean.
    * The P3 slot holds ``FIXEDAGE_k<fixed_age>`` for ablations with ``fixed_age`` (named in ``p3_policy``);
      ``gt_acc_P3`` (canonical env, mean over seeds) and ``calls_P3`` (mean total calls; int when constant)
      come from that policy. ``reference_age_variation`` is False exactly when ``fixed_age`` is set.
    """
    envmap = as_environments(envs)
    tmap = as_trajectories(trajs)
    R = _rounds(tmap, cube)
    rows: list[dict] = []
    for name, abl in plan.ablations.items():
        sched = EnvSchedule({int(k): str(v) for k, v in abl.schedule.items()})
        flags = schedule_flags(sched, envmap, R)
        age = int(abl.fixed_age) if abl.fixed_age is not None else int(plan.t4.age)
        after = list(flags["envs_after_change"])
        if after and not pairs_df.empty:
            m = (pd.to_numeric(pairs_df["age"]).astype(int) == age) & pairs_df["env"].astype(str).isin(after)
            sel = pd.to_numeric(pairs_df.loc[m, "inflation"]).astype(float)
        else:
            sel = pd.Series(dtype=float)
        third = parse_policy(f"FIXEDAGE_k{int(abl.fixed_age)}" if abl.fixed_age is not None else "P3").name
        runs = simulate_all(
            cube,
            tmap,
            plan,
            envmap,
            n_dev,
            extractor_tags=extractor_tags,
            policies=["P1", "P1b", third],
            schedule=sched,
        )
        by_pol: dict[str, list[PolicyRun]] = {}
        for r in runs:
            by_pol.setdefault(r.policy.name, []).append(r)
        f1, f1s, a1 = _far_stats(by_pol.get("P1", []))
        f1b, f1bs, a1b = _far_stats(by_pol.get("P1b", []))
        f3, f3s, a3 = _far_stats(by_pol.get(third, []))
        third_runs = by_pol.get(third, [])
        rows.append(
            {
                "ablation": str(name),
                "description": abl.description,
                "schedule": format_schedule(sched),
                "decoding_changes": bool(flags["decoding_changes"]),
                "extraction_changes": bool(flags["extraction_changes"]),
                "reference_age_variation": abl.fixed_age is None,
                "inflation_age": age,
                "inflation_envs": ",".join(after),
                "n_pairs": int(len(sel)),
                "drift_inflation_pp": float(sel.mean() * 100.0) if len(sel) else float("nan"),
                "far_P1": f1,
                "far_P1b": f1b,
                "far_P3": f3,
                "p3_policy": third,
                "far_P1_seed_mean": f1s,
                "far_P1b_seed_mean": f1bs,
                "far_P3_seed_mean": f3s,
                "n_accepted_P1": a1,
                "n_accepted_P1b": a1b,
                "n_accepted_P3": a3,
                "gt_acc_P3": (
                    float(np.mean([r.final_acc_canonical for r in third_runs]))
                    if third_runs
                    else float("nan")
                ),
                "calls_P3": _mean_calls(third_runs),
            }
        )
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype=float) for c in ABLATION_COLUMNS})
    return pd.DataFrame(rows, columns=list(ABLATION_COLUMNS))


def randomize_schedules(
    R: int,
    n: int,
    n_changes: int,
    seed: int,
    env_pool: Sequence[str] = ("E2", "E3", "E4"),
    base: str = "E1",
) -> list[dict[int, str]]:
    """``n`` random schedules (deterministic in ``seed``; EXPLORATORY).

    Each schedule starts with ``{0: base}`` and has exactly ``n_changes`` change rounds sampled without
    replacement from ``1..R``; each segment's env is drawn uniformly from ``env_pool`` minus the previous
    segment's env, so every change round is a real change. Schedules are drawn independently (repeats are
    possible); the first ``m`` schedules do not depend on ``n``.
    """
    R, n, n_changes = int(R), int(n), int(n_changes)
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if not 0 <= n_changes <= R:
        raise ValueError(f"n_changes must be in [0, R={R}], got {n_changes}")
    pool = [str(e) for e in env_pool]
    if n_changes and not pool:
        raise ValueError("env_pool is empty")
    rng = np.random.default_rng(rng_seed("schedule-randomization", int(seed), R, n_changes, base, *pool))
    out: list[dict[int, str]] = []
    for _ in range(n):
        rounds = sorted(int(r) for r in rng.choice(np.arange(1, R + 1), size=n_changes, replace=False))
        sched: dict[int, str] = {0: str(base)}
        prev = str(base)
        for r in rounds:
            choices = [e for e in pool if e != prev]
            if not choices:
                raise ValueError(f"env_pool {pool} has no environment different from {prev!r}")
            prev = choices[int(rng.integers(len(choices)))]
            sched[r] = prev
        out.append(sched)
    return out


def schedule_randomization_frame(
    cube: Cube,
    trajs: Mapping[int, Trajectory] | Sequence[Trajectory],
    plan,
    envs: Mapping[str, Environment] | None,
    n_dev: int,
    policies: Sequence[str] = ("P1", "P2", "P3", "P5"),
    extractor_tags: Mapping[str, str] | None = None,
    *,
    n: int | None = None,
) -> pd.DataFrame:
    """EXPLORATORY: one row per (schedule_id, seed, policy) over random schedules.

    Schedules come from :func:`randomize_schedules` with ``plan.schedule_randomization`` (``n`` overrides the
    count), ``R`` = the trajectories' rounds, ``base`` = the plan schedule's round-0 env and ``env_pool`` = the
    other environments of ``envs``. Columns :data:`SCHEDULE_RANDOM_COLUMNS`; ``far`` is NaN for a seed with no
    accepts; ``gt_acc`` is the final canonical-env accuracy. ``df.attrs["exploratory"]`` is True.
    """
    envmap = as_environments(envs)
    tmap = as_trajectories(trajs)
    cfg = plan.schedule_randomization
    base = str(plan.schedule[0])
    pool = tuple(e for e in envmap if e != base)
    count = int(cfg.n if n is None else n)
    R = min((t.R for t in tmap.values()), default=cube.R)
    scheds = randomize_schedules(R, count, int(cfg.n_changes), int(cfg.seed), env_pool=pool, base=base)
    rows: list[dict] = []
    for sid, ch in enumerate(scheds):
        sched = EnvSchedule(ch)
        runs = simulate_all(
            cube,
            tmap,
            plan,
            envmap,
            n_dev,
            extractor_tags=extractor_tags,
            policies=list(policies),
            schedule=sched,
        )
        text = format_schedule(ch)
        changes = ",".join(str(r) for r in sched.change_rounds())
        for r in runs:
            rows.append(
                {
                    "schedule_id": sid,
                    "schedule": text,
                    "change_rounds": changes,
                    "seed": r.seed,
                    "policy": r.policy.name,
                    "far": r.far,
                    "n_accepted": r.n_accepted,
                    "n_false_accepts": r.n_false_accepts,
                    "calls_total": r.total_calls,
                    "reference_calls": r.reference_calls,
                    "refreshes": r.refreshes,
                    "rescores": r.rescores,
                    "gt_acc": r.final_acc_canonical,
                }
            )
    if rows:
        df = pd.DataFrame(rows, columns=list(SCHEDULE_RANDOM_COLUMNS))
    else:
        df = pd.DataFrame({c: pd.Series(dtype=float) for c in SCHEDULE_RANDOM_COLUMNS})
    df.attrs["exploratory"] = True
    return df


__all__ = [
    "ABLATION_COLUMNS",
    "SCHEDULE_RANDOM_COLUMNS",
    "format_schedule",
    "randomize_schedules",
    "run_ablations",
    "schedule_flags",
    "schedule_randomization_frame",
]
