"""Paired item bootstrap.

Every seed and every pair is evaluated on the SAME eval items (the first 200 GSM8K test questions), so the
item is the resampling unit: one resampled index vector per replicate is applied to ALL per-item arrays
(every pair, every seed, every group being compared). Arrays themselves are never resampled, so each seed
keeps its fixed weight in every replicate (the seed strata are preserved), and statistics of different
groups computed from the same ``(n_items, B, seed)`` share their replicates, which makes contrasts paired.

The statistic of one replicate is ``mean over arrays of stat(array[idx])``; intervals are percentile
intervals over the B replicates.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

Stat = Callable[..., object]

_NAN3 = (float("nan"), float("nan"), float("nan"))


def bootstrap_indices(n_items: int, B: int, seed: int) -> np.ndarray:
    """(B, n_items) int64 matrix of item indices resampled with replacement (``default_rng(seed)``)."""
    if n_items < 0 or B < 0:
        raise ValueError(f"n_items and B must be >= 0, got {n_items}, {B}")
    rng = np.random.default_rng(seed)
    if n_items == 0 or B == 0:
        return np.zeros((B, n_items), dtype=np.int64)
    return rng.integers(0, n_items, size=(B, n_items), dtype=np.int64)


def _stack(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Stack per-item vectors into an (A, N) float matrix; all arrays must cover the same items."""
    mats = [np.asarray(a, dtype=float).reshape(-1) for a in arrays]
    lengths = {m.shape[0] for m in mats}
    if len(lengths) > 1:
        raise ValueError(f"paired bootstrap needs arrays over the same items; got lengths {sorted(lengths)}")
    return np.vstack(mats) if mats else np.zeros((0, 0))


def _row_stat(stat: Stat, mat: np.ndarray) -> np.ndarray:
    """Apply ``stat`` to every row of ``mat`` (B, N) -> (B,)."""
    if stat is np.mean:
        return mat.mean(axis=1)
    try:
        out = np.asarray(stat(mat, axis=1), dtype=float)
        if out.shape == (mat.shape[0],):
            return out
    except TypeError:
        pass
    return np.array([float(stat(row)) for row in mat], dtype=float)


def _point(stat: Stat, mat: np.ndarray) -> float:
    return float(np.mean([float(stat(row)) for row in mat]))


def _check_indices(idx: np.ndarray, n_items: int) -> np.ndarray:
    """Validate a (B, n_items) replicate index matrix: a resample must have exactly ``n_items`` valid items."""
    idx = np.asarray(idx)
    if idx.ndim != 2 or idx.shape[1] != n_items:
        raise ValueError(f"indices of shape {idx.shape} do not fit {n_items} items (expected (B, {n_items}))")
    if idx.size and (idx.min() < 0 or idx.max() >= n_items):
        raise ValueError(f"indices must lie in [0, {n_items}); got [{idx.min()}, {idx.max()}]")
    return idx


def bootstrap_replicates(
    arrays: Sequence[np.ndarray],
    B: int,
    seed: int,
    stat: Stat = np.mean,
    *,
    indices: np.ndarray | None = None,
) -> np.ndarray:
    """(B,) replicate statistics: mean over arrays of ``stat(array[idx_b])`` for each replicate b.

    ``indices`` (B, N) overrides ``(B, seed)``, e.g. to share replicates across calls explicitly.
    Returns an empty array for empty input.
    """
    mat = _stack(arrays)
    if mat.shape[0] == 0 or mat.shape[1] == 0:
        return np.zeros(0, dtype=float)
    n_items = mat.shape[1]
    idx = bootstrap_indices(n_items, B, seed) if indices is None else _check_indices(indices, n_items)
    if idx.shape[0] == 0:
        return np.zeros(0, dtype=float)
    if stat is np.mean:
        # mean_a mean(a[idx]) == mean(mean_a(a)[idx]) when all arrays cover the same items
        return mat.mean(axis=0)[idx].mean(axis=1)
    return np.mean([_row_stat(stat, row[idx]) for row in mat], axis=0)


def percentile_interval(reps: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile interval of replicate statistics (NaN-aware; NaNs when no finite replicate)."""
    r = np.asarray(reps, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return (float("nan"), float("nan"))
    lo, hi = np.quantile(r, [alpha / 2.0, 1.0 - alpha / 2.0])
    return (float(lo), float(hi))


def paired_bootstrap_ci(
    arrays: Sequence[np.ndarray],
    B: int,
    seed: int,
    stat: Stat = np.mean,
    alpha: float = 0.05,
    *,
    indices: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """(point, lo, hi) of ``mean over arrays of stat(array)`` with a paired item-bootstrap percentile CI.

    Each array is a per-item vector over the same items; one resampled index vector per replicate is
    applied to all arrays. Empty input (no arrays or zero items) gives NaNs; ``B == 0`` gives a NaN CI.
    """
    mat = _stack(arrays)
    if mat.shape[0] == 0 or mat.shape[1] == 0:
        return _NAN3
    point = float(mat.mean()) if stat is np.mean else _point(stat, mat)
    reps = bootstrap_replicates(list(mat), B, seed, stat, indices=indices)
    lo, hi = percentile_interval(reps, alpha)
    return (point, lo, hi)


def bootstrap_diff_ci(
    arrays_a: Sequence[np.ndarray],
    arrays_b: Sequence[np.ndarray],
    B: int,
    seed: int,
    stat: Stat = np.mean,
    alpha: float = 0.05,
    *,
    indices: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """(point, lo, hi) of S(a) - S(b), S = mean over arrays of stat, with ONE shared index vector per replicate.

    Both groups must cover the same items (e.g. H1: inflation item diffs under E2 vs under E1). Identical
    groups therefore give a zero-width interval at 0. NaNs when either group is empty.
    """
    ma, mb = _stack(arrays_a), _stack(arrays_b)
    if ma.shape[0] == 0 or mb.shape[0] == 0 or ma.shape[1] == 0 or mb.shape[1] == 0:
        return _NAN3
    if ma.shape[1] != mb.shape[1]:
        raise ValueError(f"groups cover different item counts: {ma.shape[1]} vs {mb.shape[1]}")
    pa = float(ma.mean()) if stat is np.mean else _point(stat, ma)
    pb = float(mb.mean()) if stat is np.mean else _point(stat, mb)
    idx = bootstrap_indices(ma.shape[1], B, seed) if indices is None else _check_indices(indices, ma.shape[1])
    ra = bootstrap_replicates(list(ma), B, seed, stat, indices=idx)
    rb = bootstrap_replicates(list(mb), B, seed, stat, indices=idx)
    lo, hi = percentile_interval(ra - rb, alpha)
    return (pa - pb, lo, hi)
