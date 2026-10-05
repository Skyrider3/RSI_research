"""Tests for the factorial all-pairs drift analysis (hand-built cubes; no DB)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from driftlab.analysis.allpairs import (
    PAIR_COLUMNS,
    SUMMARY_COLUMNS,
    T4_COLUMNS,
    T7_COLUMNS,
    PairResult,
    all_pairs,
    env_axes,
    filter_pairs,
    pivot_grid,
    summarize_pairs,
    table4_frame,
    table7_frame,
)
from driftlab.analysis.cube import empty_cube, make_trajectory, set_cell
from driftlab.analysis.metrics import mean_sd, wilson
from driftlab.config import AnalysisPlan, ExperimentConfig
from driftlab.environments import build_environments
from driftlab.keys import rng_seed

ENVS = {"E1": ("greedy", "v1"), "E2": ("t02", "v1"), "E3": ("greedy", "v2"), "E4": ("t02", "v2")}
PLAN = AnalysisPlan()


def _plan(**kw) -> AnalysisPlan:
    return AnalysisPlan.model_validate(kw)


def _draws(R: int, G: int = 1) -> list[tuple[str, int]]:
    return [("round", r) for r in range(R + 1)] + [("gt", g) for g in range(G)]


def _outputs(rng: np.random.Generator, q: float, N: int) -> tuple[np.ndarray, np.ndarray]:
    """(v1, v2) correctness of one generation; the lenient v2 is a superset of the strict v1."""
    v1 = rng.random(N) < q
    v2 = v1 | (rng.random(N) < 0.3)
    return v1, v2


def synth_cube(
    R: int = 4,
    N: int = 24,
    seeds: tuple[int, ...] = (0,),
    physical_greedy: bool = False,
    flip: float = 0.2,
    tag: str = "base",
):
    """Full-triangle cube. Greedy reruns are cache hits (shared gen rows) unless ``physical_greedy``;
    every t02 round is an independent draw; one ("gt", 0) draw per t02 slot."""
    cube = empty_cube("test", seeds, ["greedy", "t02"], ["v1", "v2"], R + 1, _draws(R), N)
    for s in seeds:
        rng = np.random.default_rng(rng_seed("test-allpairs", tag, s))
        quality = rng.uniform(0.25, 0.75, size=R + 1)
        for k in range(R + 1):
            v1, v2 = _outputs(rng, quality[k], N)
            set_cell(cube, s, "greedy", k, ("round", k), {"v1": v1, "v2": v2}, physical=True)
            created_rows = cube.gen_rows(s, "greedy", k, ("round", k)).copy()
            for r in range(k + 1, R + 1):
                if physical_greedy:
                    f = rng.random(N) < flip
                    w1 = v1 ^ f
                    w2 = (v2 ^ f) | w1
                    set_cell(cube, s, "greedy", k, ("round", r), {"v1": w1, "v2": w2}, physical=True)
                else:
                    set_cell(cube, s, "greedy", k, ("round", r), {"v1": v1, "v2": v2}, gen_rows=created_rows)
            for r in range(k, R + 1):
                t1, t2 = _outputs(rng, quality[k], N)
                trunc = rng.random(N) < 0.1
                set_cell(
                    cube, s, "t02", k, ("round", r), {"v1": t1, "v2": t2}, physical=True, truncated=trunc
                )
            g1, g2 = _outputs(rng, quality[k], N)
            set_cell(cube, s, "t02", k, ("gt", 0), {"v1": g1, "v2": g2}, physical=True)
    return cube


def trajs_for(seeds, R: int, advanced=None) -> dict:
    adv = advanced if advanced is not None else [False] + [t % 2 == 1 for t in range(1, R + 1)]
    return {s: make_trajectory(s, [f"seed{s}-p{k}" for k in range(R + 1)], adv) for s in seeds}


def _ones(N: int, idx) -> np.ndarray:
    v = np.zeros(N, dtype=np.int8)
    v[list(idx)] = 1
    return v


# --------------------------------------------------------------------------- env axes


def test_env_axes_accepts_environments_tuples_and_config_sections():
    from_env = env_axes(build_environments())
    assert from_env == ENVS
    assert env_axes(ENVS) == ENVS
    assert env_axes(ExperimentConfig().environments) == ENVS
    assert env_axes(None) == ENVS
    with pytest.raises(TypeError):
        env_axes({"E9": 3})


def test_environment_objects_and_tuples_give_identical_pairs():
    cube, trajs = synth_cube(), trajs_for([0], 4)
    a = all_pairs(cube, trajs, build_environments(), PLAN)
    b = all_pairs(cube, trajs, ENVS, PLAN)
    pd.testing.assert_frame_equal(a.df, b.df)
    assert list(a.df.columns) == list(PAIR_COLUMNS)
    assert len(a.df) == 4 * (2 + 3 + 4 + 5)  # 4 envs x sum_{j=1..4} (j + 1)
    assert not a.skipped


# --------------------------------------------------------------------------- (1)-(4) inflation


def test_e1_cache_hit_reruns_are_zero_by_construction():
    cube, trajs = synth_cube(physical_greedy=False), trajs_for([0], 4)
    pr = all_pairs(cube, trajs, ENVS, PLAN, env_ids=["E1"])
    df = pr.df
    assert (df["inflation"] == 0.0).all()
    assert (df["w_stored"] == df["w_rerun"]).all()
    assert df["by_construction"].all()
    assert (df["same_gen_frac"] == 1.0).all()
    assert not df.loc[df["age"] > 0, "physical_rerun"].any()
    assert not df["env_changed"].any()
    assert not df["flip"].any()
    assert all((d == 0).all() for d in pr.item_diffs.values())


def test_physical_rerun_that_differs_gives_measured_inflation():
    N = 20
    cube = empty_cube("t", [0], ["greedy", "t02"], ["v1", "v2"], 2, _draws(1), N)
    stored = _ones(N, range(10))
    rerun = _ones(N, range(15))
    cand = _ones(N, range(5, 20))
    set_cell(cube, 0, "greedy", 0, ("round", 0), {"v1": stored, "v2": stored}, physical=True)
    set_cell(cube, 0, "greedy", 0, ("round", 1), {"v1": rerun, "v2": rerun}, physical=True)
    set_cell(cube, 0, "greedy", 1, ("round", 1), {"v1": cand, "v2": cand}, physical=True)
    pr = all_pairs(cube, trajs_for([0], 1), ENVS, PLAN, env_ids=["E1"], ages=[1])
    row = pr.df.iloc[0]
    assert (row["j"], row["i"], row["ref_slot"], row["cur_slot"]) == (1, 0, 0, 0)
    assert (row["w_stored"], row["w_rerun"]) == (10, 5)
    assert row["inflation"] == pytest.approx(0.25)
    assert row["infl_generation"] == pytest.approx(0.25) and row["infl_extract"] == 0.0
    assert not row["by_construction"] and row["physical_rerun"]
    assert row["same_gen_frac"] == 0.0
    assert pr.item_diffs[(0, "E1", 1, 0)].sum() == 5

    # the synthetic physical cube: no E1 pair at age > 0 is flagged, and some inflation is nonzero
    cube2 = synth_cube(physical_greedy=True, flip=0.25)
    df = all_pairs(cube2, trajs_for([0], 4), ENVS, PLAN, env_ids=["E1"]).df
    older = df[df["age"] > 0]
    assert not older["by_construction"].any()
    assert older["physical_rerun"].all()
    assert (older["inflation"] != 0).any()


def test_e3_monotone_extractor_gives_nonnegative_inflation():
    cube, trajs = synth_cube(physical_greedy=False, R=6, N=30), trajs_for([0], 6)
    df = all_pairs(cube, trajs, ENVS, PLAN, env_ids=["E3"]).df
    assert (df["inflation"] >= 0).all()
    assert (df["infl_generation"] == 0).all()  # same text, same extractor
    assert (df["infl_extract"] >= 0).all()
    assert (df["inflation"] > 0).any()
    assert not df["by_construction"].any()  # extractor changed: a measured (not structural) effect
    assert (df["same_gen_frac"] == 1.0).all()
    assert df["env_changed"].all()


def test_decomposition_sums_exactly_for_every_row():
    seeds = (0, 1, 2)
    cube, trajs = synth_cube(physical_greedy=True, seeds=seeds, R=5), trajs_for(seeds, 5)
    pr = all_pairs(cube, trajs, ENVS, PLAN)
    df = pr.df
    assert len(df) == 3 * 4 * sum(j + 1 for j in range(1, 6))
    assert (df["infl_extract"] + df["infl_generation"] == df["inflation"]).all()
    np.testing.assert_allclose(df["inflation"], (df["w_stored"] - df["w_rerun"]) / df["n"], atol=1e-12)
    for r in ("stored", "rescored", "rerun", "fresh"):
        np.testing.assert_allclose(df[f"win_{r}"], df[f"w_{r}"] / df["n"])
    for row in df.itertuples():
        d = pr.item_diffs[(row.seed, row.env, row.j, row.i)]
        assert d.dtype == np.int8 and len(d) == cube.n_items
        assert int(d.sum()) == row.w_stored - row.w_rerun
    assert (df["flip"] == (df["dec_stored"] != df["dec_rerun"])).all()
    assert (df["env_changed"] == (df["env"] != "E1")).all()
    assert df.loc[df["decoding"] == "t02", "physical_rerun"].all()
    assert df.loc[df["decoding"] == "t02", "trunc_rerun"].gt(0).any()


# --------------------------------------------------------------------------- (5) reference mode


def test_incumbent_and_chain_reference_modes_pick_different_slots():
    R = 5
    adv = [False, True, False, True, False, False]  # inc_slot = [0, 0, 1, 1, 3, 3]
    trajs = trajs_for([0], R, advanced=adv)
    assert trajs[0].inc_slot == [0, 0, 1, 1, 3, 3]
    cube = synth_cube(R=R)
    inc = all_pairs(cube, trajs, ENVS, PLAN, env_ids=["E1"]).df.set_index(["j", "i"])
    chain = all_pairs(cube, trajs, ENVS, _plan(reference_mode="chain"), env_ids=["E1"]).df.set_index(
        ["j", "i"]
    )
    assert inc.loc[(5, 2), "ref_slot"] == 1 and chain.loc[(5, 2), "ref_slot"] == 2
    assert inc.loc[(5, 4), "ref_slot"] == 3 and chain.loc[(5, 4), "ref_slot"] == 4
    assert (inc["cur_slot"] == chain["cur_slot"]).all()
    for (j, i), ref in inc["ref_slot"].items():
        assert ref == trajs[0].inc_slot[i]
        assert chain.loc[(j, i), "ref_slot"] == i
    assert (inc["ref_slot"] != chain["ref_slot"]).any()


# --------------------------------------------------------------------------- (6) FAR logic


def test_far_cur_stale_stored_reference_accepts_worse_candidate():
    N = 20
    cube = empty_cube("t", [0], ["greedy", "t02"], ["v1", "v2"], 3, _draws(2), N)
    # Environment staleness: under E3 the stored v1 scores of slot 0 are strict (5/20), the same text under
    # v2 scores 16/20; the candidate (slot 1) gets 12/20 under v2.
    s0_v1, s0_v2 = _ones(N, range(5)), _ones(N, range(16))
    c1 = _ones(N, range(8, 20))
    for r in (0, 1, 2):
        rows = None if r == 0 else cube.gen_rows(0, "greedy", 0, ("round", 0)).copy()
        set_cell(
            cube, 0, "greedy", 0, ("round", r), {"v1": s0_v1, "v2": s0_v2}, gen_rows=rows, physical=r == 0
        )
    set_cell(cube, 0, "greedy", 1, ("round", 1), {"v1": c1, "v2": c1}, physical=True)
    pr = all_pairs(cube, trajs_for([0], 1), ENVS, PLAN, env_ids=["E3"], ages=[1])
    row = pr.df.iloc[0]
    assert (row["w_stored"], row["l_stored"]) == (12, 5)
    assert row["dec_stored"] and not row["dec_rerun"] and not row["dec_fresh"]
    assert row["gt_cand"] == pytest.approx(0.6) and row["gt_cur"] == pytest.approx(0.8)
    assert row["fa_cur_stored"] and not row["fa_cur_rerun"] and not row["fa_cur_fresh"]
    assert row["flip"]
    s = summarize_pairs(pr, B=200)
    assert s.loc[0, "far_cur_stored"] == 1.0 and s.loc[0, "n_accept_stored"] == 1
    assert math.isnan(s.loc[0, "far_cur_rerun"]) and s.loc[0, "n_accept_rerun"] == 0

    # Prompt staleness (E1, deterministic greedy): candidate 1 advanced, so inc_slot[2] = 1; the reference
    # created at round 0 (slot 0, weak) accepts candidate 2, which is worse than the current incumbent.
    cube2 = empty_cube("t", [0], ["greedy", "t02"], ["v1", "v2"], 3, _draws(2), N)
    acc = {0: range(5), 1: range(15), 2: range(10)}
    for k in range(3):
        for r in range(k, 3):
            v = _ones(N, acc[k])
            set_cell(cube2, 0, "greedy", k, ("round", r), {"v1": v, "v2": v}, physical=True)
    trajs = trajs_for([0], 2, advanced=[False, True, False])
    df = all_pairs(cube2, trajs, ENVS, PLAN, env_ids=["E1"]).df.set_index(["j", "i"])
    r20 = df.loc[(2, 0)]
    assert (r20["ref_slot"], r20["cur_slot"]) == (0, 1)
    assert r20["dec_stored"] and r20["dec_rerun"] and not r20["dec_fresh"]
    assert r20["fa_cur_stored"] and r20["fa_cur_rerun"] and not r20["fa_cur_fresh"]
    assert not r20["fa_ref_stored"]  # candidate IS better than the (stale) reference prompt
    r22 = df.loc[(2, 2)]  # age 0: the reference is the current incumbent -> rejected
    assert r22["ref_slot"] == 1 and not r22["dec_stored"] and not r22["fa_cur_stored"]


# --------------------------------------------------------------------------- (7) split half


def test_split_half_uses_disjoint_item_sets():
    N, R = 24, 3
    cube = empty_cube("t", [0], ["greedy", "t02"], ["v1", "v2"], R + 1, _draws(R), N)
    for k in range(R + 1):
        # slot k is correct on the first half iff k is odd, and on k+1 items of the second half
        v = np.zeros(N, dtype=np.int8)
        if k % 2:
            v[: N // 2] = 1
        v[N // 2 : N // 2 + k + 1] = 1
        for r in range(k, R + 1):
            set_cell(cube, 0, "greedy", k, ("round", r), {"v1": v, "v2": v}, physical=True)
            set_cell(cube, 0, "t02", k, ("round", r), {"v1": v, "v2": v}, physical=True)
        set_cell(cube, 0, "t02", k, ("gt", 0), {"v1": v, "v2": v}, physical=True)
    trajs = trajs_for([0], R, advanced=[False] * (R + 1))
    pr = all_pairs(cube, trajs, ENVS, _plan(gt={"mode": "split_half"}))
    assert np.array_equal(pr.item_index, np.arange(N // 2))
    assert pr.gt_index is not None and np.array_equal(pr.gt_index, np.arange(N // 2, N))
    assert not set(pr.item_index) & set(pr.gt_index)
    assert (pr.df["n"] == N // 2).all()
    assert all(len(d) == N // 2 for d in pr.item_diffs.values())
    for row in pr.df.itertuples():
        assert row.gt_cand == pytest.approx((row.j + 1) / (N - N // 2))
        assert row.gt_cur == pytest.approx(1 / (N - N // 2))  # incumbent stays slot 0
        assert row.w_fresh == (N // 2 if row.j % 2 else 0)
    full = all_pairs(cube, trajs, ENVS, PLAN)
    assert np.array_equal(full.item_index, np.arange(N)) and full.gt_index is None
    assert (full.df["n"] == N).all()


# --------------------------------------------------------------------------- (8) ages + missing cells


def test_ages_filter_and_missing_cells_are_skipped_not_raised():
    cube = synth_cube(R=3)
    pr = all_pairs(cube, trajs_for([0], 3), ENVS, PLAN, ages=[0, 3])
    assert set(pr.df["age"]) == {0, 3}
    assert len(pr.df) == 4 * (3 + 1)  # age 0: j = 1..3, age 3: j = 3

    # trajectory longer than the cube (built for fewer rounds): extra pairs are skipped
    pr = all_pairs(cube, trajs_for([0], 5), ENVS, PLAN, env_ids=["E1"])
    assert len(pr.df) == sum(j + 1 for j in range(1, 4))
    assert len(pr.skipped) == sum(j + 1 for j in (4, 5))
    assert all(len(t) == 5 and isinstance(t[4], str) for t in pr.skipped)

    # a single missing cell (slot 2 never rerun at round 3 under greedy): in chain mode only (j=3, i=2) needs it
    cube.correct[0, 0, 2, 3] = -1
    flat = trajs_for([0], 3, advanced=[False, False, False, False])
    pr = all_pairs(cube, flat, ENVS, _plan(reference_mode="chain"), env_ids=["E1"])
    assert (0, "E1", 3, 2) not in pr.item_diffs
    assert [t[:4] for t in pr.skipped] == [(0, "E1", 3, 2)]
    assert len(pr.df) == sum(j + 1 for j in range(1, 4)) - 1
    assert not all_pairs(cube, flat, ENVS, PLAN, env_ids=["E1"]).skipped  # incumbent mode never needs it

    # no GT draws for the sampling decoding -> E2/E4 pairs skipped with a reason
    cube3 = synth_cube(R=2)
    cube3.correct[:, 1, :, cube3.draws.index(("gt", 0))] = -1
    pr = all_pairs(cube3, trajs_for([0], 2), ENVS, PLAN)
    assert set(pr.df["env"]) == {"E1", "E3"}
    assert pr.skipped and all("GT" in t[4] for t in pr.skipped)

    # seed without a trajectory, and an all-filtered result
    pr = all_pairs(synth_cube(seeds=(0, 1)), trajs_for([0], 4), ENVS, PLAN, env_ids=["E1"])
    assert (1, None, None, None, "no trajectory for seed") in pr.skipped
    empty = all_pairs(cube, trajs_for([0], 3), ENVS, PLAN, ages=[99])
    assert empty.df.empty and list(empty.df.columns) == list(PAIR_COLUMNS)
    s = summarize_pairs(empty)
    assert s.empty and list(s.columns) == ["env", "age", *SUMMARY_COLUMNS]

    with pytest.raises(ValueError):
        all_pairs(cube, trajs_for([0], 3), ENVS, PLAN, env_ids=["E9"])


def test_missing_cell_skip_records_the_exact_pair():
    cube = synth_cube(R=3)
    cube.correct[0, 0, 0, 2] = -1  # greedy slot 0 at round 2: needed as rerun of (j=2, i<=1) and stored (i=2)
    trajs = trajs_for([0], 3, advanced=[False, False, False, False])  # incumbent stays slot 0
    pr = all_pairs(cube, trajs, ENVS, PLAN, env_ids=["E1"])
    skipped_pairs = {t[2:4] for t in pr.skipped}
    assert skipped_pairs == {(2, 0), (2, 1), (2, 2), (3, 2)}
    assert set(zip(pr.df["j"], pr.df["i"], strict=True)) & skipped_pairs == set()


# --------------------------------------------------------------------------- (9) summaries


@pytest.fixture(scope="module")
def three_seed_pairs() -> PairResult:
    seeds = (0, 1, 2)
    cube = synth_cube(R=5, N=30, seeds=seeds, physical_greedy=True, flip=0.3, tag="summary")
    return all_pairs(cube, trajs_for(seeds, 5), ENVS, PLAN)


def test_summarize_pairs_pooled_wilson_bootstrap_and_seed_stats(three_seed_pairs):
    pr = three_seed_pairs
    s = summarize_pairs(pr, B=500, seed=11)
    assert list(s.columns) == ["env", "age", *SUMMARY_COLUMNS]
    assert len(s) == 4 * 6
    for row in s.itertuples():
        g = pr.df[(pr.df["env"] == row.env) & (pr.df["age"] == row.age)]
        assert row.n_pairs == len(g) and row.n_seeds == g["seed"].nunique()
        assert row.inflation == pytest.approx((g["w_stored"] - g["w_rerun"]).sum() / g["n"].sum())
        assert row.inflation == pytest.approx(row.infl_extract + row.infl_generation, abs=1e-12)
        assert row.win_stored == pytest.approx(g["w_stored"].sum() / g["n"].sum())
        assert row.infl_lo - 1e-12 <= row.inflation <= row.infl_hi + 1e-12
        # Wilson on pooled counts
        k, n = int(g["fa_cur_stored"].sum()), int(g["dec_stored"].sum())
        p, lo, hi = wilson(k, n)
        assert (row.n_fa_cur_stored, row.n_accept_stored) == (k, n)
        for a, b in ((row.far_cur_stored, p), (row.far_cur_stored_lo, lo), (row.far_cur_stored_hi, hi)):
            assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b)
        if n:  # metrics.wilson(0, n) can return lo ~ 1e-17
            assert lo - 1e-12 <= p <= hi + 1e-12
        # per-seed mean +- sd
        per_seed = [gs["inflation"].mean() for _, gs in g.groupby("seed")]
        m, sd, _ = mean_sd(per_seed)
        assert row.infl_seed_mean == pytest.approx(m) and row.infl_seed_sd == pytest.approx(sd)
        ws = [gs["win_rerun"].mean() for _, gs in g.groupby("seed")]
        assert row.win_rerun_seed_mean == pytest.approx(np.mean(ws))
        far_seed = [
            gs["fa_cur_stored"].sum() / gs["dec_stored"].sum() if gs["dec_stored"].sum() else np.nan
            for _, gs in g.groupby("seed")
        ]
        fm, fsd, fn = mean_sd(far_seed)
        if fn:
            assert row.far_cur_stored_seed_mean == pytest.approx(fm)
            assert row.far_cur_stored_seed_sd == pytest.approx(fsd)
        else:
            assert math.isnan(row.far_cur_stored_seed_mean)
        assert row.flip_rate == pytest.approx(g["flip"].mean())
    # the E1 age-0 cell is identical by construction; changed-env cells are not
    e1a0 = s[(s["env"] == "E1") & (s["age"] == 0)].iloc[0]
    assert e1a0["all_by_construction"] and e1a0["by_construction_frac"] == 1.0
    assert not s.loc[s["env"] != "E1", "all_by_construction"].any()


def test_summarize_pairs_hand_checked_seed_mean_sd():
    rows, diffs = [], {}
    base = dict.fromkeys(PAIR_COLUMNS, 0)
    for seed, infl_counts in ((0, (2, 4)), (1, (6, 6)), (2, (0, 2))):
        for j, c in enumerate(infl_counts, start=1):
            r = dict(base, seed=seed, env="E4", j=j, i=0, age=j, n=10, w_stored=c, w_rescored=c, w_rerun=0)
            r.update(inflation=c / 10, infl_extract=0.0, infl_generation=c / 10, by_construction=False)
            r.update(flip=False, trunc_rerun=0.0, dec_stored=True, fa_cur_stored=seed == 1)
            r.update(win_stored=c / 10, win_rerun=0.0)
            rows.append(r)
            diffs[(seed, "E4", j, 0)] = np.array([1] * c + [0] * (10 - c), dtype=np.int8)
    pr = PairResult(df=pd.DataFrame(rows), item_diffs=diffs, item_index=np.arange(10))
    s = summarize_pairs(pr, by=(), B=300).iloc[0]
    # per-seed means of inflation: 0.3, 0.6, 0.1
    assert s["infl_seed_mean"] == pytest.approx(1.0 / 3)
    assert s["infl_seed_sd"] == pytest.approx(np.std([0.3, 0.6, 0.1], ddof=1))
    assert s["inflation"] == pytest.approx(20 / 60)
    assert s["n_pairs"] == 6 and s["n_seeds"] == 3
    assert s["far_cur_stored"] == pytest.approx(2 / 6)
    assert s["far_cur_stored_seed_mean"] == pytest.approx(1 / 3)
    assert s["far_cur_stored_seed_sd"] == pytest.approx(np.std([0, 1, 0], ddof=1))
    assert s["infl_lo"] <= s["inflation"] <= s["infl_hi"]


def test_filter_pairs_and_pooled_single_group(three_seed_pairs):
    pr = three_seed_pairs
    sub = filter_pairs(pr, env_ids=["E2", "E4"], ages=[3])
    assert set(sub.df["env"]) == {"E2", "E4"} and set(sub.df["age"]) == {3}
    assert set(sub.item_diffs) == {(r.seed, r.env, r.j, r.i) for r in sub.df.itertuples()}
    one = summarize_pairs(sub, by=(), B=200)
    assert len(one) == 1 and one.loc[0, "n_pairs"] == len(sub.df)
    assert list(one.columns) == list(SUMMARY_COLUMNS)


def test_summary_is_deterministic(three_seed_pairs):
    a = summarize_pairs(three_seed_pairs, B=300, seed=5)
    b = summarize_pairs(three_seed_pairs, B=300, seed=5)
    pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------------------------- (10) paper tables


def test_table7_frame_has_ten_rows_with_missing_cells_as_nan(three_seed_pairs):
    s = summarize_pairs(three_seed_pairs, B=200)
    t7 = table7_frame(s, PLAN)
    assert list(t7.columns) == list(T7_COLUMNS)
    assert len(t7) == 10
    assert list(t7["age"]) == [0, 0, 1, 1, 3, 3, 5, 5, 10, 10]
    assert list(t7["env_status"][:2]) == ["Unchanged (E1)", "Changed (E4)"]
    assert list(t7["env"][:2]) == ["E1", "E4"]
    age10 = t7[t7["age"] == 10]  # R = 5: no pair is 10 rounds old
    assert (age10["n_pairs"] == 0).all() and age10["inflation"].isna().all()
    assert age10["far_cur_stored"].isna().all() and not age10["by_construction"].any()
    a5 = t7[(t7["age"] == 5) & (t7["env"] == "E4")].iloc[0]
    assert a5["n_pairs"] == 3  # one pair (j=5, i=0) per seed
    a0 = t7[(t7["age"] == 0) & (t7["env"] == "E1")].iloc[0]
    assert a0["by_construction"] and a0["inflation"] == 0.0

    empty = table7_frame(summarize_pairs(filter_pairs(three_seed_pairs, ages=[99])), PLAN)
    assert len(empty) == 10 and (empty["n_pairs"] == 0).all()


def test_table4_frame_rows_and_labels(three_seed_pairs):
    s = summarize_pairs(three_seed_pairs, B=200)
    t4 = table4_frame(s, PLAN)
    assert list(t4.columns) == list(T4_COLUMNS)
    assert list(t4["env"]) == ["E1", "E2", "E3", "E4"]
    assert list(t4["env_label"]) == ["Unchanged", "Decoding change", "Extraction change", "Multiple changes"]
    assert (t4["age"] == 3).all() and (t4["n_pairs"] == 3 * 3).all()
    np.testing.assert_allclose(t4["infl_extract"] + t4["infl_generation"], t4["inflation"], atol=1e-12)
    grid = pivot_grid(s)
    assert list(grid.columns) == ["E1", "E2", "E3", "E4"] and list(grid.index) == list(range(6))
