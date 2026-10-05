"""Tests for the paired item bootstrap."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from driftlab.analysis.bootstrap import (
    bootstrap_diff_ci,
    bootstrap_indices,
    bootstrap_replicates,
    paired_bootstrap_ci,
    percentile_interval,
)
from driftlab.keys import rng_seed


def _arrays(n_arrays: int, n_items: int, tag: str = "a") -> list[np.ndarray]:
    rng = np.random.default_rng(rng_seed("test-bootstrap", tag))
    return [rng.integers(-1, 2, size=n_items).astype(np.int8) for _ in range(n_arrays)]


def test_bootstrap_indices_shape_range_and_determinism():
    idx = bootstrap_indices(30, 200, seed=7)
    assert idx.shape == (200, 30)
    assert idx.min() >= 0 and idx.max() < 30
    assert np.array_equal(idx, bootstrap_indices(30, 200, seed=7))
    assert not np.array_equal(idx, bootstrap_indices(30, 200, seed=8))
    assert bootstrap_indices(0, 5, seed=1).shape == (5, 0)
    assert bootstrap_indices(4, 0, seed=1).shape == (0, 4)
    with pytest.raises(ValueError):
        bootstrap_indices(-1, 5, seed=1)


def test_paired_ci_is_deterministic_and_contains_point():
    arrs = _arrays(6, 40)
    a = paired_bootstrap_ci(arrs, B=1000, seed=20261004)
    b = paired_bootstrap_ci(arrs, B=1000, seed=20261004)
    assert a == b
    point, lo, hi = a
    assert point == pytest.approx(np.mean([x.mean() for x in arrs]))
    assert lo <= point <= hi and lo < hi
    assert paired_bootstrap_ci(arrs, B=1000, seed=1) != a


def test_one_index_vector_per_replicate_is_shared_by_all_arrays():
    arrs = _arrays(3, 25, "shared")
    B, seed = 50, 3
    idx = bootstrap_indices(25, B, seed)
    expected = np.array([np.mean([x[idx[b]].mean() for x in arrs]) for b in range(B)])
    close = {"rtol": 1e-12, "atol": 1e-12}
    np.testing.assert_allclose(bootstrap_replicates(arrs, B, seed), expected, **close)
    # the general (non-mean) path applies the same rows
    general = bootstrap_replicates(arrs, B, seed, stat=lambda v: float(np.mean(v)))
    np.testing.assert_allclose(general, expected, **close)
    np.testing.assert_allclose(bootstrap_replicates(arrs, B, seed, indices=idx), expected, **close)


def test_identical_groups_give_zero_width_difference_ci():
    arrs = _arrays(4, 30, "diff")
    assert bootstrap_diff_ci(arrs, arrs, B=500, seed=9) == (0.0, 0.0, 0.0)
    shifted = [x.astype(float) + 0.5 for x in arrs]
    point, lo, hi = bootstrap_diff_ci(shifted, arrs, B=500, seed=9)
    assert point == pytest.approx(0.5) and lo == pytest.approx(0.5) and hi == pytest.approx(0.5)
    # independent CIs of the two groups are wide, but the paired difference is exact
    _, alo, ahi = paired_bootstrap_ci(arrs, B=500, seed=9)
    assert ahi - alo > 0


def test_diff_ci_matches_replicate_difference():
    a, b = _arrays(3, 20, "x"), _arrays(2, 20, "y")
    point, lo, hi = bootstrap_diff_ci(a, b, B=400, seed=4)
    ra = bootstrap_replicates(a, 400, 4)
    rb = bootstrap_replicates(b, 400, 4)
    assert (lo, hi) == percentile_interval(ra - rb)
    assert point == pytest.approx(np.mean([x.mean() for x in a]) - np.mean([x.mean() for x in b]))
    with pytest.raises(ValueError):
        bootstrap_diff_ci(a, _arrays(1, 21), B=10, seed=1)


def test_empty_and_degenerate_inputs():
    nan3 = paired_bootstrap_ci([], B=100, seed=1)
    assert all(math.isnan(v) for v in nan3)
    assert all(math.isnan(v) for v in paired_bootstrap_ci([np.array([])], B=100, seed=1))
    assert all(math.isnan(v) for v in bootstrap_diff_ci([], _arrays(1, 5), B=100, seed=1))
    point, lo, hi = paired_bootstrap_ci(_arrays(2, 5), B=0, seed=1)
    assert math.isfinite(point) and math.isnan(lo) and math.isnan(hi)
    assert bootstrap_replicates([], 10, 1).shape == (0,)
    assert all(math.isnan(v) for v in percentile_interval(np.array([np.nan, np.nan])))
    const = [np.ones(12), np.ones(12)]
    assert paired_bootstrap_ci(const, B=200, seed=2) == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        paired_bootstrap_ci([np.ones(3), np.ones(4)], B=10, seed=1)


def test_non_mean_statistic():
    arrs = [np.arange(11, dtype=float), np.arange(11, dtype=float) * 2]
    point, lo, hi = paired_bootstrap_ci(arrs, B=300, seed=5, stat=np.median)
    assert point == pytest.approx((5 + 10) / 2)
    assert lo <= point <= hi


@settings(max_examples=30, deadline=None)
@given(
    data=st.lists(st.lists(st.integers(-1, 1), min_size=8, max_size=8), min_size=1, max_size=5),
    seed=st.integers(0, 2**31 - 1),
)
def test_property_point_inside_ci_bounds(data, seed):
    arrs = [np.asarray(row, dtype=np.int8) for row in data]
    point, lo, hi = paired_bootstrap_ci(arrs, B=200, seed=seed)
    assert -1.0 <= lo <= hi <= 1.0
    assert point == pytest.approx(float(np.mean(arrs)))
