"""Factorial all-pairs drift analysis (docs/ARCHITECTURE.md section 6.2): the source of T4, T7 and T8.

For every seed s, current environment c = (d_c, x_c), candidate j in 1..R and reference creation round
i in 0..j (age = j - i), with the storage environment st = plan.storage_env = (d_st, x_st):

* ``ref      = inc_slot[i]`` (``reference_mode: incumbent``) or ``i`` (``chain``); ``cur = inc_slot[j]``
* ``stored   = V(s, d_st, ref, ("round", i), x_st)``  scores kept from storage time
* ``rescored = V(s, d_st, ref, ("round", i), x_c)``   same text, current extractor
* ``rerun    = V(s, d_c,  ref, ("round", j), x_c)``   same prompt regenerated under c at round j
* ``fresh    = V(s, d_c,  cur, ("round", j), x_c)``   current incumbent under c
* ``cand     = V(s, d_c,  j,   ("round", j), x_c)``

Counts ``w_R = sum(cand & ~R)``, ``l_R = sum(~cand & R)`` over the decision items; decisions use the plan's
promotion rule; ground truth uses ``cube.gt_acc`` under c at round j. ``inflation`` is defined as
``infl_extract + infl_generation`` so that the decomposition is exact in floating point; it equals
``(w_stored - w_rerun) / n`` up to one ulp. Pairs whose inflation is zero *by construction* (the rerun is a
cache hit of the stored generation and the extractor did not change) are flagged ``by_construction`` and
must never be reported as a measured zero; ``gen_by_construction`` flags the same cache hit for the
generation part alone (e.g. E3 without physical reruns). ``gt_coupled`` marks pairs whose GT reuses the
decision cells and items (greedy, or ``same_draw``): there ``fa_cur_fresh`` and ``fa_ref_rerun`` cannot be
True (any rule needing wins > losses), which is the documented greedy caveat, not a finding.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from driftlab.analysis.bootstrap import bootstrap_indices, paired_bootstrap_ci
from driftlab.analysis.cube import DRAW_ROUND, Cube, MissingCell, Trajectory
from driftlab.analysis.metrics import decide, mean_sd, wilson

REFS: tuple[str, ...] = ("stored", "rescored", "rerun", "fresh")
GT_EPS = 1e-9  # tolerance for "gt_cand <= gt_x" (GT accuracies are means of per-item rates)

PAIR_COLUMNS: tuple[str, ...] = (
    "seed",
    "env",
    "j",
    "i",
    "age",
    "ref_slot",
    "cur_slot",
    "n",
    *(f"{p}_{r}" for r in REFS for p in ("w", "l")),
    *(f"win_{r}" for r in REFS),
    "inflation",
    "infl_extract",
    "infl_generation",
    *(f"dec_{r}" for r in REFS),
    "flip",
    "gt_cand",
    "gt_cur",
    "gt_ref",
    *(f"fa_cur_{r}" for r in REFS),
    "fa_ref_stored",
    "fa_ref_rescored",
    "fa_ref_rerun",
    "fa_ref_fresh",
    "by_construction",
    "same_gen_frac",
    "physical_rerun",
    "trunc_rerun",
    "env_changed",
    "decoding",
    "extractor",
    "gen_by_construction",
    "gt_coupled",
)

_FAR_REFS: tuple[str, ...] = ("stored", "rescored", "rerun", "fresh")

SUMMARY_COLUMNS: tuple[str, ...] = (
    "n_pairs",
    "n_seeds",
    "win_stored",
    "win_rescored",
    "win_rerun",
    "win_fresh",
    "win_stored_seed_mean",
    "win_stored_seed_sd",
    "win_rerun_seed_mean",
    "win_rerun_seed_sd",
    "inflation",
    "infl_lo",
    "infl_hi",
    "infl_seed_mean",
    "infl_seed_sd",
    "infl_extract",
    "infl_generation",
    "flip_rate",
    *(
        col
        for r in _FAR_REFS
        for col in (
            f"n_accept_{r}",
            f"n_fa_cur_{r}",
            f"far_cur_{r}",
            f"far_cur_{r}_lo",
            f"far_cur_{r}_hi",
            f"far_cur_{r}_seed_mean",
            f"far_cur_{r}_seed_sd",
        )
    ),
    "far_ref_stored",
    "far_ref_rerun",
    "by_construction_frac",
    "all_by_construction",
    "trunc_rerun",
    "gen_by_construction_frac",
    "all_gen_by_construction",
    "gt_coupled_frac",
)

T4_LABELS: dict[str, str] = {
    "E1": "Unchanged",
    "E2": "Decoding change",
    "E3": "Extraction change",
    "E4": "Multiple changes",
}

T4_COLUMNS: tuple[str, ...] = (
    "env",
    "env_label",
    "age",
    "n_pairs",
    "n_seeds",
    "stored_win",
    "rerun_win",
    "inflation",
    "infl_lo",
    "infl_hi",
    "infl_extract",
    "infl_generation",
    "flip_rate",
    "by_construction",
    "by_construction_frac",
    "stored_win_seed_mean",
    "stored_win_seed_sd",
    "rerun_win_seed_mean",
    "rerun_win_seed_sd",
    "infl_seed_mean",
    "infl_seed_sd",
    "gen_by_construction",
)

T7_COLUMNS: tuple[str, ...] = (
    "age",
    "env",
    "env_status",
    "n_pairs",
    "stored_win",
    "rerun_win",
    "inflation",
    "infl_lo",
    "infl_hi",
    "far_cur_stored",
    "far_cur_rerun",
    "far_cur_fresh",
    "flip_rate",
    "n_accept_stored",
    "by_construction",
    "by_construction_frac",
    "n_seeds",
    "far_cur_stored_lo",
    "far_cur_stored_hi",
    "gt_coupled",
)

PairKey = tuple[int, str, int, int]


@dataclass
class PairResult:
    """Output of :func:`all_pairs`.

    * ``df``: one row per (seed, env, j, i); columns :data:`PAIR_COLUMNS`.
    * ``item_diffs``: (seed, env, j, i) -> int8 per-item vector ``(cand & ~stored) - (cand & ~rerun)`` over
      the decision items (its mean is the pair's inflation).
    * ``item_index``: eval item indices the decisions used (all items, or ``[0, N//2)`` under split_half).
    * ``skipped``: ``(seed, env, j, i, reason)`` for pairs skipped because a cell was missing.
    * ``gt_index``: item indices ground truth used (``None`` = all items).
    """

    df: pd.DataFrame
    item_diffs: dict[PairKey, np.ndarray]
    item_index: np.ndarray
    skipped: list[tuple] = field(default_factory=list)
    gt_index: np.ndarray | None = None


class _Missing(Exception):
    """A cell needed by one pair is absent from the cube."""


# --------------------------------------------------------------------------- environments


def env_axes(envs: Mapping[str, object] | None) -> dict[str, tuple[str, str]]:
    """``env_id -> (decoding_id, extractor)``.

    Accepts ``dict[str, Environment]``, ``dict[str, (decoding_id, extractor[, description])]`` or objects with
    ``decoding`` (an id or a ``Decoding``) and ``extractor`` attributes (e.g. config ``EnvSection``).
    ``None`` means :data:`driftlab.environments.DEFAULT_ENVIRONMENTS`.
    """
    if envs is None:
        from driftlab.environments import DEFAULT_ENVIRONMENTS

        envs = DEFAULT_ENVIRONMENTS
    out: dict[str, tuple[str, str]] = {}
    for eid, e in envs.items():
        if isinstance(e, (tuple, list)):
            if len(e) < 2:
                raise ValueError(f"environment {eid!r} must be (decoding_id, extractor), got {e!r}")
            out[str(eid)] = (str(e[0]), str(e[1]))
        elif hasattr(e, "decoding") and hasattr(e, "extractor"):
            dec = e.decoding
            out[str(eid)] = (str(getattr(dec, "id", dec)), str(e.extractor))
        else:
            raise TypeError(f"cannot read decoding/extractor of environment {eid!r}: {e!r}")
    return out


# --------------------------------------------------------------------------- cube access with skip semantics


def _vec(cube: Cube, seed: int, decoding: str, slot: int, draw: tuple[str, int], x: str) -> np.ndarray:
    if not 0 <= slot < cube.n_slots:
        raise _Missing(f"slot {slot} outside cube (n_slots={cube.n_slots})")
    try:
        return cube.vec(seed, decoding, slot, draw, x)
    except (MissingCell, KeyError, IndexError) as e:
        raise _Missing(f"missing cell seed={seed} {decoding} slot={slot} draw={draw} {x}") from e


def _gt(
    cube: Cube, seed: int, decoding: str, slot: int, round_: int, x: str, mode: str, items: np.ndarray | None
) -> float:
    if not 0 <= slot < cube.n_slots:
        raise _Missing(f"GT slot {slot} outside cube (n_slots={cube.n_slots})")
    try:
        return cube.gt_acc(seed, decoding, slot, round_=round_, extractor=x, mode=mode, items=items)
    except (MissingCell, KeyError, IndexError) as e:
        raise _Missing(
            f"missing GT cell seed={seed} {decoding} slot={slot} round={round_} {x} ({mode})"
        ) from e


def _same_gen_frac(cube: Cube, a: tuple, b: tuple, items: np.ndarray | None) -> float:
    """``cube.same_gen_frac`` restricted to the decision items (identical when all items are used)."""
    if items is None:
        return cube.same_gen_frac(a, b)
    ga, gb = cube.gen_rows(*a)[items], cube.gen_rows(*b)[items]
    ok = (ga >= 0) & (gb >= 0)
    if not ok.any():
        return float("nan")
    return float((ga[ok] == gb[ok]).mean())


def _same_generation(cube: Cube, stored_cell: tuple, rerun_cell: tuple, sgf: float) -> bool:
    """True when the rerun cell is (on every compared item) the stored generation itself.

    The identical cell (age 0 under the storage decoding) is always the same generation. Otherwise the gen-row
    fraction decides; when gen rows are unavailable (NaN fraction) fall back to the matrix semantics of
    section 4: a non-physical cell of the same decoding is a cache hit of the slot's creation generation,
    so the pair is flagged rather than reported as a measured zero.
    """
    if stored_cell == rerun_cell:
        return True
    if np.isfinite(sgf):
        return bool(sgf == 1.0)
    _, d_st, ref, rd_i = stored_cell
    d_c = rerun_cell[1]
    if d_st != d_c or cube.is_physical(*rerun_cell):
        return False
    return rd_i == (DRAW_ROUND, ref) or not cube.is_physical(*stored_cell)


def _item_sets(n_items: int, gt_mode: str) -> tuple[np.ndarray, np.ndarray | None, str]:
    """(decision items, GT items or None for all, mode passed to cube.gt_acc)."""
    if gt_mode == "split_half":
        h = n_items // 2
        return np.arange(h), np.arange(h, n_items), "independent_draw"
    return np.arange(n_items), None, gt_mode


def _rate(k: float, n: float) -> float:
    return float(k) / float(n) if n else float("nan")


# --------------------------------------------------------------------------- all pairs


def _pair(
    cube: Cube,
    traj: Trajectory,
    seed: int,
    env: str,
    axes_c: tuple[str, str],
    axes_st: tuple[str, str],
    storage_env: str,
    j: int,
    i: int,
    plan,
    dec_items: np.ndarray,
    gt_items: np.ndarray | None,
    gt_mode: str,
) -> tuple[dict, np.ndarray]:
    d_c, x_c = axes_c
    d_st, x_st = axes_st
    if j >= len(traj.inc_slot) or i >= len(traj.inc_slot):
        raise _Missing(f"trajectory of seed {seed} has no round {max(i, j)}")
    ref = i if plan.reference_mode == "chain" else int(traj.inc_slot[i])
    cur = int(traj.inc_slot[j])
    rd_i, rd_j = (DRAW_ROUND, i), (DRAW_ROUND, j)
    split = gt_items is not None
    sel = dec_items if split else slice(None)

    vecs = {
        "stored": _vec(cube, seed, d_st, ref, rd_i, x_st)[sel],
        "rescored": _vec(cube, seed, d_st, ref, rd_i, x_c)[sel],
        "rerun": _vec(cube, seed, d_c, ref, rd_j, x_c)[sel],
        "fresh": _vec(cube, seed, d_c, cur, rd_j, x_c)[sel],
    }
    cand = _vec(cube, seed, d_c, j, rd_j, x_c)[sel]
    gt_cand = _gt(cube, seed, d_c, j, j, x_c, gt_mode, gt_items)
    gt_cur = _gt(cube, seed, d_c, cur, j, x_c, gt_mode, gt_items)
    gt_ref = _gt(cube, seed, d_c, ref, j, x_c, gt_mode, gt_items)

    n = int(len(cand))
    w = {r: int((cand & ~v).sum()) for r, v in vecs.items()}
    lo = {r: int((~cand & v).sum()) for r, v in vecs.items()}
    dec = {r: bool(decide(w[r], lo[r], n, plan.promotion_rule)) for r in REFS}
    worse_cur = gt_cand <= gt_cur + GT_EPS
    worse_ref = gt_cand <= gt_ref + GT_EPS

    infl_extract = _rate(w["stored"] - w["rescored"], n)
    infl_generation = _rate(w["rescored"] - w["rerun"], n)
    stored_cell = (seed, d_st, ref, rd_i)
    rerun_cell = (seed, d_c, ref, rd_j)
    sgf = _same_gen_frac(cube, stored_cell, rerun_cell, dec_items if split else None)
    same_gen = _same_generation(cube, stored_cell, rerun_cell, sgf)
    # GT read from the very cells (and items) the decisions used: then w - l = n * (gt_cand - gt_x) for the
    # fresh (and same-cell rerun) reference, so fa_cur_fresh / fa_ref_rerun are False by construction.
    gt_coupled = (not split) and (cube.is_greedy(d_c) or gt_mode == "same_draw")

    row: dict = {
        "seed": int(seed),
        "env": str(env),
        "j": int(j),
        "i": int(i),
        "age": int(j - i),
        "ref_slot": ref,
        "cur_slot": cur,
        "n": n,
    }
    for r in REFS:
        row[f"w_{r}"] = w[r]
        row[f"l_{r}"] = lo[r]
    for r in REFS:
        row[f"win_{r}"] = _rate(w[r], n)
    row["inflation"] = infl_extract + infl_generation
    row["infl_extract"] = infl_extract
    row["infl_generation"] = infl_generation
    for r in REFS:
        row[f"dec_{r}"] = dec[r]
    row["flip"] = dec["stored"] != dec["rerun"]
    row["gt_cand"] = gt_cand
    row["gt_cur"] = gt_cur
    row["gt_ref"] = gt_ref
    for r in REFS:
        row[f"fa_cur_{r}"] = bool(dec[r] and worse_cur)
    for r in REFS:
        row[f"fa_ref_{r}"] = bool(dec[r] and worse_ref)
    row["by_construction"] = bool(same_gen and x_st == x_c)
    row["same_gen_frac"] = sgf
    row["physical_rerun"] = cube.is_physical(*rerun_cell)
    row["trunc_rerun"] = cube.trunc_rate(*rerun_cell)
    row["env_changed"] = str(env) != str(storage_env)
    row["decoding"] = d_c
    row["extractor"] = x_c
    row["gen_by_construction"] = bool(same_gen)  # infl_generation == 0 by construction (cache hit)
    row["gt_coupled"] = bool(gt_coupled)

    diff = (cand & ~vecs["stored"]).astype(np.int8) - (cand & ~vecs["rerun"]).astype(np.int8)
    return row, diff


def all_pairs(
    cube: Cube,
    trajs: Mapping[int, Trajectory] | Sequence[Trajectory],
    envs: Mapping[str, object] | None,
    plan,
    env_ids: Sequence[str] | None = None,
    ages: Sequence[int] | None = None,
) -> PairResult:
    """Score every (seed, env, j, i) pair; see the module docstring for the definitions.

    ``plan`` is a :class:`driftlab.config.AnalysisPlan` (uses ``reference_mode``, ``storage_env``,
    ``promotion_rule`` and ``gt.mode``). ``envs`` maps env ids to environments (see :func:`env_axes`);
    ``env_ids`` restricts the current environments (default: all of ``envs``); ``ages`` keeps only pairs with
    ``j - i`` in ``ages``. Pairs that need a cell the cube does not have are skipped and listed in
    ``PairResult.skipped``; this function never raises for missing cells. ``trajs`` may also be a sequence of
    trajectories (keyed by ``Trajectory.seed``).
    """
    if not isinstance(trajs, Mapping):
        trajs = {int(t.seed): t for t in trajs}
    env_ids, ages = _as_list(env_ids), _as_list(ages)
    axes = env_axes(envs)
    st = str(plan.storage_env)
    if st not in axes:
        raise ValueError(f"storage env {st!r} is not among the environments {sorted(axes)}")
    targets = list(dict.fromkeys(str(e) for e in (env_ids if env_ids is not None else list(axes))))
    unknown = [e for e in targets if e not in axes]
    if unknown:
        raise ValueError(f"unknown environment ids {unknown}; known: {sorted(axes)}")
    age_set = None if ages is None else {int(a) for a in ages}
    dec_items, gt_items, gt_mode = _item_sets(cube.n_items, plan.gt.mode)

    rows: list[dict] = []
    diffs: dict[PairKey, np.ndarray] = {}
    skipped: list[tuple] = []
    for seed in cube.seeds:
        traj = trajs.get(seed)
        if traj is None:
            skipped.append((int(seed), None, None, None, "no trajectory for seed"))
            continue
        for env in targets:
            for j in range(1, traj.R + 1):
                for i in range(j + 1):
                    if age_set is not None and (j - i) not in age_set:
                        continue
                    try:
                        row, diff = _pair(
                            cube,
                            traj,
                            seed,
                            env,
                            axes[env],
                            axes[st],
                            st,
                            j,
                            i,
                            plan,
                            dec_items,
                            gt_items,
                            gt_mode,
                        )
                    except _Missing as e:
                        skipped.append((int(seed), env, j, i, str(e)))
                        continue
                    rows.append(row)
                    diffs[(int(seed), env, j, i)] = diff
    df = pd.DataFrame(rows, columns=list(PAIR_COLUMNS)) if rows else _empty_pairs()
    return PairResult(df=df, item_diffs=diffs, item_index=dec_items, skipped=skipped, gt_index=gt_items)


_BOOL_PAIR_COLUMNS = (
    "flip",
    "by_construction",
    "physical_rerun",
    "env_changed",
    "gen_by_construction",
    "gt_coupled",
)


def _empty_pairs() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype=float) for c in PAIR_COLUMNS})
    for c in ("seed", "j", "i", "age", "ref_slot", "cur_slot", "n"):
        df[c] = df[c].astype(np.int64)
    for c in PAIR_COLUMNS:
        if c.startswith(("dec_", "fa_")) or c in _BOOL_PAIR_COLUMNS:
            df[c] = df[c].astype(bool)
    for c in ("env", "decoding", "extractor"):
        df[c] = df[c].astype(str)
    return df


def _as_list(v: object) -> list | None:
    """``None`` -> ``None``; a bare string or scalar -> ``[v]`` (never iterate a string's characters)."""
    if v is None:
        return None
    if isinstance(v, (str, bytes, int, np.integer)):
        return [v]
    return list(v)


def filter_pairs(
    pr: PairResult,
    *,
    env_ids: Sequence[str] | None = None,
    ages: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
) -> PairResult:
    """Subset of a PairResult (rows and their item diffs), e.g. the envs an ablation schedule visits."""
    env_ids, ages, seeds = _as_list(env_ids), _as_list(ages), _as_list(seeds)
    m = pd.Series(True, index=pr.df.index)
    if env_ids is not None:
        m &= pr.df["env"].isin([str(e) for e in env_ids])
    if ages is not None:
        m &= pr.df["age"].isin([int(a) for a in ages])
    if seeds is not None:
        m &= pr.df["seed"].isin([int(s) for s in seeds])
    df = pr.df.loc[m].reset_index(drop=True)
    keys = set(_row_keys(df))
    return PairResult(
        df=df,
        item_diffs={k: v for k, v in pr.item_diffs.items() if k in keys},
        item_index=pr.item_index,
        skipped=list(pr.skipped),
        gt_index=pr.gt_index,
    )


def _row_keys(df: pd.DataFrame) -> list[PairKey]:
    return [
        (int(s), str(e), int(j), int(i))
        for s, e, j, i in zip(df["seed"], df["env"], df["j"], df["i"], strict=True)
    ]


# --------------------------------------------------------------------------- summaries


def _wilson(k: int, n: int) -> tuple[float, float, float]:
    """``metrics.wilson`` with exact bounds at the edges (k == 0 -> lo = 0, k == n -> hi = 1)."""
    p, lo, hi = wilson(k, n)
    if n > 0 and k <= 0:
        lo = 0.0
    if n > 0 and k >= n:
        hi = 1.0
    return p, lo, hi


def _frac(g: pd.DataFrame, col: str) -> float:
    return float(g[col].astype(bool).mean()) if len(g) and col in g.columns else float("nan")


def _far(g: pd.DataFrame, ref: str) -> tuple[int, int, tuple[float, float, float], float, float]:
    n_acc = int(g[f"dec_{ref}"].sum())
    n_fa = int(g[f"fa_cur_{ref}"].sum())
    per_seed = [
        _rate(int(gs[f"fa_cur_{ref}"].sum()), int(gs[f"dec_{ref}"].sum())) for _, gs in g.groupby("seed")
    ]
    m, sd, _ = mean_sd(per_seed)
    return n_acc, n_fa, _wilson(n_fa, n_acc), m, sd


def _seed_mean_sd(g: pd.DataFrame, col: str) -> tuple[float, float]:
    m, sd, _ = mean_sd([float(gs[col].mean()) for _, gs in g.groupby("seed")])
    return m, sd


def _summarize_group(
    g: pd.DataFrame, pr: PairResult, B: int, seed: int, indices: np.ndarray | None
) -> dict[str, object]:
    nsum = int(g["n"].sum())
    out: dict[str, object] = {"n_pairs": int(len(g)), "n_seeds": int(g["seed"].nunique())}
    for r in REFS:
        out[f"win_{r}"] = _rate(int(g[f"w_{r}"].sum()), nsum)
    out["win_stored_seed_mean"], out["win_stored_seed_sd"] = _seed_mean_sd(g, "win_stored")
    out["win_rerun_seed_mean"], out["win_rerun_seed_sd"] = _seed_mean_sd(g, "win_rerun")
    ext = _rate(int((g["w_stored"] - g["w_rescored"]).sum()), nsum)
    gen = _rate(int((g["w_rescored"] - g["w_rerun"]).sum()), nsum)
    arrays = [pr.item_diffs[k] for k in _row_keys(g) if k in pr.item_diffs]
    _, lo, hi = paired_bootstrap_ci(arrays, B, seed, indices=indices) if arrays else (np.nan,) * 3
    out["inflation"] = ext + gen
    out["infl_lo"], out["infl_hi"] = lo, hi
    out["infl_seed_mean"], out["infl_seed_sd"] = _seed_mean_sd(g, "inflation")
    out["infl_extract"], out["infl_generation"] = ext, gen
    out["flip_rate"] = float(g["flip"].mean()) if len(g) else float("nan")
    for r in _FAR_REFS:
        n_acc, n_fa, (p, plo, phi), sm, ssd = _far(g, r)
        out[f"n_accept_{r}"], out[f"n_fa_cur_{r}"] = n_acc, n_fa
        out[f"far_cur_{r}"], out[f"far_cur_{r}_lo"], out[f"far_cur_{r}_hi"] = p, plo, phi
        out[f"far_cur_{r}_seed_mean"], out[f"far_cur_{r}_seed_sd"] = sm, ssd
    for r in ("stored", "rerun"):
        out[f"far_ref_{r}"] = _rate(int(g[f"fa_ref_{r}"].sum()), int(g[f"dec_{r}"].sum()))
    out["by_construction_frac"] = float(g["by_construction"].mean()) if len(g) else float("nan")
    out["all_by_construction"] = bool(len(g) > 0 and g["by_construction"].all())
    out["trunc_rerun"] = float(g["trunc_rerun"].mean()) if len(g) else float("nan")
    out["gen_by_construction_frac"] = _frac(g, "gen_by_construction")
    out["all_gen_by_construction"] = bool(out["gen_by_construction_frac"] == 1.0)
    out["gt_coupled_frac"] = _frac(g, "gt_coupled")
    return out


def _check_item_diffs(pr: PairResult) -> int:
    """Number of decision items; every row must have an item diff of that length (or no row has one)."""
    keys = _row_keys(pr.df)
    have = [k in pr.item_diffs for k in keys]
    if any(have) and not all(have):
        missing = [k for k, h in zip(keys, have, strict=True) if not h][:3]
        raise ValueError(f"item_diffs missing for {have.count(False)} of {len(keys)} rows, e.g. {missing}")
    n_dec = len(pr.item_index)
    bad = {len(pr.item_diffs[k]) for k, h in zip(keys, have, strict=True) if h} - {n_dec}
    if bad:
        raise ValueError(f"item_diffs have lengths {sorted(bad)} but item_index has {n_dec} items")
    return n_dec


def summarize_pairs(
    pr: PairResult,
    by: Sequence[str] = ("env", "age"),
    B: int = 2000,
    seed: int = 20261004,
) -> pd.DataFrame:
    """Per-group summary of an all-pairs result (columns: ``by`` + :data:`SUMMARY_COLUMNS`).

    Win rates, inflation and its parts are pooled (sum of counts / sum of n). ``infl_lo/hi`` is a 95%
    percentile CI from the paired item bootstrap over the group's item diffs; every group uses the same
    replicate index vectors (``bootstrap_indices(n, B, seed)``), so groups are coupled like the H1 contrasts.
    FAR = sum(fa_cur) / sum(dec) with a Wilson 95% CI (NaN with no accepts); ``*_seed_mean/_seed_sd`` are the
    mean and sd over seeds of per-seed values (NaN-aware). ``by=()`` summarizes all rows as one group.
    ``gen_by_construction_frac`` is the share of pairs whose generation part is zero by construction (cache
    hit) and ``gt_coupled_frac`` the share whose GT reuses the decision cells (FAR_cur(fresh) and
    FAR_ref(rerun) are then 0 by construction). Raises ``ValueError`` when ``item_diffs`` do not match the
    rows or ``item_index`` (a CI over other items than the point estimate would be silently wrong).
    """
    by = [str(b) for b in (_as_list(by) or [])]
    missing = [b for b in by if b not in pr.df.columns]
    if missing:
        raise ValueError(f"cannot group by unknown columns {missing}")
    cols = [*by, *SUMMARY_COLUMNS]
    if pr.df.empty:
        return pd.DataFrame({c: pd.Series(dtype=float) for c in cols})
    n_dec = _check_item_diffs(pr)
    indices = bootstrap_indices(n_dec, B, seed) if n_dec and B > 0 else None
    rows: list[dict] = []
    if not by:
        rows.append(_summarize_group(pr.df, pr, B, seed, indices))
    else:
        for key, g in pr.df.groupby(by, sort=True):
            key = key if isinstance(key, tuple) else (key,)
            row = dict(zip(by, key, strict=True))
            row.update(_summarize_group(g, pr, B, seed, indices))
            rows.append(row)
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- paper tables


def _lookup(summary: pd.DataFrame, env: str, age: int) -> pd.Series | None:
    for c in ("env", "age"):
        if c not in summary.columns:
            raise ValueError(f"summary must be grouped by ('env', 'age'); missing column {c!r}")
    if summary.empty:
        return None
    m = (summary["env"].astype(str) == str(env)) & (summary["age"].astype(int) == int(age))
    hit = summary.loc[m]
    return None if hit.empty else hit.iloc[0]


def _get(row: pd.Series | None, col: str, default: object = float("nan")) -> object:
    if row is None or col not in row.index:
        return default
    return row[col]


def _flag(row: pd.Series | None, col: str) -> bool:
    """A boolean summary flag; missing or NaN counts as False (``bool(nan)`` would be True)."""
    v = _get(row, col, False)
    return False if pd.isna(v) else bool(v)


def _all(row: pd.Series | None, frac_col: str) -> bool:
    v = _get(row, frac_col)
    return False if pd.isna(v) else bool(float(v) == 1.0)


def table7_frame(summary: pd.DataFrame, plan) -> pd.DataFrame:
    """T7: reference age (``plan.t7.ages``) x {unchanged, changed} env; long form, age-major.

    ``by_construction`` is True when every pair of the cell is identical by construction (render as "‡").
    ``gt_coupled`` is True when every pair's GT reuses the decision cells (greedy, or ``same_draw``): then
    ``far_cur_fresh`` is 0 by construction and must carry the greedy-coupling caveat, not read as a finding.
    Cells without pairs keep their row with ``n_pairs = 0`` and NaN metrics.
    """
    t7 = plan.t7
    statuses = [(str(t7.unchanged_env), f"Unchanged ({t7.unchanged_env})")]
    statuses.append((str(t7.changed_env), f"Changed ({t7.changed_env})"))
    rows = []
    for age in t7.ages:
        for env, label in statuses:
            r = _lookup(summary, env, age)
            rows.append(
                {
                    "age": int(age),
                    "env": env,
                    "env_status": label,
                    "n_pairs": int(_get(r, "n_pairs", 0)),
                    "stored_win": _get(r, "win_stored"),
                    "rerun_win": _get(r, "win_rerun"),
                    "inflation": _get(r, "inflation"),
                    "infl_lo": _get(r, "infl_lo"),
                    "infl_hi": _get(r, "infl_hi"),
                    "far_cur_stored": _get(r, "far_cur_stored"),
                    "far_cur_rerun": _get(r, "far_cur_rerun"),
                    "far_cur_fresh": _get(r, "far_cur_fresh"),
                    "flip_rate": _get(r, "flip_rate"),
                    "n_accept_stored": int(_get(r, "n_accept_stored", 0)),
                    "by_construction": _flag(r, "all_by_construction"),
                    "by_construction_frac": _get(r, "by_construction_frac"),
                    "n_seeds": int(_get(r, "n_seeds", 0)),
                    "far_cur_stored_lo": _get(r, "far_cur_stored_lo"),
                    "far_cur_stored_hi": _get(r, "far_cur_stored_hi"),
                    "gt_coupled": _all(r, "gt_coupled_frac"),
                }
            )
    return pd.DataFrame(rows, columns=list(T7_COLUMNS))


def table4_frame(
    summary: pd.DataFrame,
    plan,
    env_ids: Sequence[str] = ("E1", "E2", "E3", "E4"),
    labels: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """T4: one row per environment at reference age ``plan.t4.age`` (labels from :data:`T4_LABELS`).

    ``gen_by_construction`` is True when every pair's generation part is 0 by construction (cache-hit rerun,
    e.g. E3 without physical greedy reruns): render that part as "‡", not as a measured 0.
    """
    labels = dict(T4_LABELS if labels is None else labels)
    age = int(plan.t4.age)
    rows = []
    for env in env_ids:
        r = _lookup(summary, env, age)
        rows.append(
            {
                "env": str(env),
                "env_label": labels.get(str(env), str(env)),
                "age": age,
                "n_pairs": int(_get(r, "n_pairs", 0)),
                "n_seeds": int(_get(r, "n_seeds", 0)),
                "stored_win": _get(r, "win_stored"),
                "rerun_win": _get(r, "win_rerun"),
                "inflation": _get(r, "inflation"),
                "infl_lo": _get(r, "infl_lo"),
                "infl_hi": _get(r, "infl_hi"),
                "infl_extract": _get(r, "infl_extract"),
                "infl_generation": _get(r, "infl_generation"),
                "flip_rate": _get(r, "flip_rate"),
                "by_construction": _flag(r, "all_by_construction"),
                "by_construction_frac": _get(r, "by_construction_frac"),
                "stored_win_seed_mean": _get(r, "win_stored_seed_mean"),
                "stored_win_seed_sd": _get(r, "win_stored_seed_sd"),
                "rerun_win_seed_mean": _get(r, "win_rerun_seed_mean"),
                "rerun_win_seed_sd": _get(r, "win_rerun_seed_sd"),
                "infl_seed_mean": _get(r, "infl_seed_mean"),
                "infl_seed_sd": _get(r, "infl_seed_sd"),
                "gen_by_construction": _all(r, "gen_by_construction_frac"),
            }
        )
    return pd.DataFrame(rows, columns=list(T4_COLUMNS))


def pivot_grid(summary: pd.DataFrame, value: str = "inflation") -> pd.DataFrame:
    """Age x env grid of one summary column (T7b / the dashboard heatmap)."""
    if summary.empty:
        return pd.DataFrame()
    return summary.pivot(index="age", columns="env", values=value).sort_index()
