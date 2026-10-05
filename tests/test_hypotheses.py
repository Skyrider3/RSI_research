"""Pre-registered hypotheses H1-H3 on hand-built cubes with known effects (no DB)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from driftlab.analysis.allpairs import all_pairs, summarize_pairs
from driftlab.analysis.cube import empty_cube, make_trajectory, set_cell
from driftlab.analysis.hypotheses import (
    HYPOTHESIS_COLUMNS,
    MIN_ACCEPTS,
    _far_diff,
    _memo_envs,
    _PolicyStats,
    _pool,
    evaluate_hypotheses,
    h2_rows,
    h3_rows,
)
from driftlab.analysis.policies import simulate, simulate_all
from driftlab.config import AnalysisPlan
from driftlab.environments import build_environments
from driftlab.extraction import extractor_tags
from driftlab.keys import rng_seed

ENVS = build_environments()
SEEDS = (0, 1, 2)


def _draws(R: int, G: int = 1) -> list[tuple[str, int]]:
    return [("round", r) for r in range(R + 1)] + [("gt", g) for g in range(G)]


def factorial_cube(R: int = 11, N: int = 200, seeds=SEEDS, identical: bool = False, extra_v2: float = 0.2):
    """Full-triangle cube. Greedy reruns are cache hits of the creation generation; t02 rounds are independent
    draws; v2 = v1 | extra (lenient extractor). ``identical``: every decoding/extractor/draw holds the greedy
    creation outputs (stored == rerun everywhere, so no inflation can exist)."""
    cube = empty_cube("h", seeds, ["greedy", "t02"], ["v1", "v2"], R + 1, _draws(R), N)
    for s in seeds:
        rng = np.random.default_rng(rng_seed("test-hypotheses", s, identical))
        quality = rng.uniform(0.35, 0.75, size=R + 1)

        def outputs(q: float, rng: np.random.Generator = rng) -> dict[str, np.ndarray]:
            v1 = rng.random(N) < q
            v2 = v1.copy() if identical else v1 | (rng.random(N) < extra_v2)
            return {"v1": v1, "v2": v2}

        for k in range(R + 1):
            created = outputs(quality[k])
            set_cell(cube, s, "greedy", k, ("round", k), created, physical=True)
            rows = cube.gen_rows(s, "greedy", k, ("round", k)).copy()
            for r in range(k + 1, R + 1):
                set_cell(cube, s, "greedy", k, ("round", r), created, gen_rows=rows)
            for r in range(k, R + 1):
                set_cell(
                    cube,
                    s,
                    "t02",
                    k,
                    ("round", r),
                    created if identical else outputs(quality[k]),
                    physical=True,
                )
            set_cell(
                cube, s, "t02", k, ("gt", 0), created if identical else outputs(quality[k]), physical=True
            )
    return cube


def trajs_for(R: int = 11, seeds=SEEDS) -> dict[int, object]:
    return {
        s: make_trajectory(
            s, [f"s{s}-p{k}" for k in range(R + 1)], [False] + [t % 2 == 1 for t in range(1, R + 1)]
        )
        for s in seeds
    }


def prefix_cube(counts: list[int], N: int = 40, seeds=SEEDS):
    """Greedy-only cube: slot k answers exactly items [0, counts[k]) (nested sets, identical under v1/v2)."""
    R = len(counts) - 1
    cube = empty_cube("p", seeds, ["greedy"], ["v1", "v2"], R + 1, _draws(R, 0), N)
    for s in seeds:
        for k, c in enumerate(counts):
            v = (np.arange(N) < c).astype(np.int8)
            set_cell(cube, s, "greedy", k, ("round", k), {"v1": v, "v2": v}, physical=True)
            rows = cube.gen_rows(s, "greedy", k, ("round", k)).copy()
            for r in range(k + 1, R + 1):
                set_cell(cube, s, "greedy", k, ("round", r), {"v1": v, "v2": v}, gen_rows=rows)
    return cube


def evaluate(cube, trajs, plan, B=400, B_policy=40, runs=None, seed=None):
    pr = all_pairs(cube, trajs, ENVS, plan)
    if runs is None:
        runs = simulate_all(
            cube, trajs, plan, ENVS, n_dev=200, policies=["P1", "P1b", "P2", "P3"], on_missing="skip"
        )
    return pr, evaluate_hypotheses(
        pr, runs, cube, trajs, plan, ENVS, n_dev=200, B=B, seed=seed, B_policy=B_policy
    )


def row(df: pd.DataFrame, hyp: str, subject: str, status: str | None = None) -> pd.Series:
    m = (df["hypothesis"] == hyp) & (df["subject"] == subject)
    if status is not None:
        m &= df["status"] == status
    sub = df.loc[m]
    assert len(sub) == 1, (hyp, subject, df[["hypothesis", "subject", "status"]])
    return sub.iloc[0]


def h1_diff(df: pd.DataFrame, env: str) -> pd.Series:
    sub = df.loc[(df["hypothesis"] == "H1") & (df["subject"] == env) & (df["status"] != "descriptive")]
    assert len(sub) == 1
    return sub.iloc[0]


# --------------------------------------------------------------------------- H1


@pytest.fixture(scope="module")
def drift_result():
    cube, trajs, plan = factorial_cube(), trajs_for(), AnalysisPlan()
    pr, df = evaluate(cube, trajs, plan)
    return cube, trajs, plan, pr, df


def test_columns_and_rows(drift_result):
    *_, df = drift_result
    assert tuple(df.columns) == HYPOTHESIS_COLUMNS
    assert list(df["hypothesis"].unique()) == ["H1", "H2", "H3"]
    h1 = df.loc[df["hypothesis"] == "H1"]
    assert list(h1["subject"]) == ["E1", "E2", "E3", "E4", "E2", "E3", "E4", "overall"]
    assert (h1.loc[h1["status"] == "descriptive", "unit"] == "pp").all()
    assert set(df.loc[df["hypothesis"] == "H2", "subject"]) == {"P1-P2", "P1b-P2"}
    assert list(df.loc[df["hypothesis"] == "H3", "subject"]) == ["P3/P2", "P3-P2", "overall"]
    assert set(df.loc[df["hypothesis"] == "H2", "unit"]) == {"pp of FAR"}


def test_h1_extraction_change_inflation_detected(drift_result):
    *_, pr, df = drift_result
    e3 = h1_diff(df, "E3")
    assert e3["status"] == "effect detected"
    assert e3["estimate"] > 0 and e3["ci_lo"] > 0
    assert e3["n"] == 3 * (11 - 3 + 1)  # matched pairs at age 3 over 3 seeds
    raw_e3 = row(df, "H1", "E3", "descriptive")
    assert raw_e3["estimate"] > 0 and raw_e3["ci_lo"] > 0
    # the control's cache-hit reruns make its inflation 0 by construction (flagged with the double dagger)
    raw_e1 = row(df, "H1", "E1", "descriptive")
    assert raw_e1["estimate"] == 0 and raw_e1["ci_lo"] == 0 and raw_e1["ci_hi"] == 0
    assert "‡" in raw_e1["note"] and "‡" in e3["note"]
    assert e3["estimate"] == pytest.approx(raw_e3["estimate"] - raw_e1["estimate"])
    assert row(df, "H1", "overall")["status"] == "supported"


def test_h1_raw_rows_reproduce_pair_summary_intervals(drift_result):
    *_, plan, pr, df = drift_result
    summ = summarize_pairs(pr, by=("env", "age"), B=400, seed=plan.bootstrap.seed)
    for env in ("E1", "E2", "E3", "E4"):
        s = summ.loc[(summ["env"] == env) & (summ["age"] == plan.t4.age)].iloc[0]
        r = row(df, "H1", env, "descriptive")
        assert r["estimate"] == pytest.approx(s["inflation"] * 100, abs=1e-9)
        assert r["ci_lo"] == pytest.approx(s["infl_lo"] * 100, abs=1e-9)
        assert r["ci_hi"] == pytest.approx(s["infl_hi"] * 100, abs=1e-9)


def test_h1_notes_state_the_direction_and_the_drivers(drift_result):
    """H1 is two-sided ('the effect may be positive or negative'): every note says which way the observed effect
    goes, and the overall note names the environments that drive the result with their signs."""
    *_, df = drift_result
    raw_e3 = row(df, "H1", "E3", "descriptive")
    assert (
        f"observed direction: stored references INFLATE measured win rates under E3 ({raw_e3['estimate']:+.2f} pp)"
        in (raw_e3["note"])
    )
    assert "leave measured win rates unchanged under E1" in row(df, "H1", "E1", "descriptive")["note"]
    e3 = h1_diff(df, "E3")
    assert f"inflation under E3 is HIGHER than under the E1 control by {e3['estimate']:.2f} pp" in e3["note"]
    overall = row(df, "H1", "overall")["note"]
    assert "two-sided test" in overall
    detected = [e for e in ("E2", "E3", "E4") if h1_diff(df, e)["status"] == "effect detected"]
    undetected = [e for e in ("E2", "E3", "E4") if h1_diff(df, e)["status"] == "no effect detected"]
    assert "E3" in detected and undetected  # this cube: extraction drift detected, pure sampling drift not
    driven = overall.split("result driven by ")[1].split("; CI includes 0 under ")[0]
    for e in detected:
        assert f"{e} {h1_diff(df, e)['estimate']:+.2f} pp vs E1 (stored references INFLATE" in driven, e
    rest = overall.split("; CI includes 0 under ")[1]
    for e in undetected:
        assert f"{e} {h1_diff(df, e)['estimate']:+.2f} pp vs E1" in rest, e


def test_h1_deflation_is_an_effect_too_and_is_named():
    """A stricter re-scoring extractor makes stored references DEFLATE measured win rates: the two-sided test
    detects it (status rules unchanged) and the notes say DEFLATE / LOWER."""
    cube, trajs, plan = factorial_cube(), trajs_for(), AnalysisPlan()
    S, D, K, R, _, N = cube.correct.shape
    s, d, k, n = np.ix_(range(S), range(D), range(K), range(N))
    keep = np.broadcast_to(
        ((s * 1000003 + d * 10007 + k * 101 + n * 7919) % 5 != 0)[:, :, :, None, :], (S, D, K, R, N)
    )
    v1 = cube.correct[..., cube.extractors.index("v1"), :]
    cube.correct[..., cube.extractors.index("v2"), :] = np.where(
        v1 >= 0, ((v1 == 1) & keep).astype(np.int8), -1
    )
    _, df = evaluate(cube, trajs, plan, B=200, B_policy=5)
    e3 = h1_diff(df, "E3")
    assert e3["status"] == "effect detected" and e3["ci_hi"] < 0
    assert "inflation under E3 is LOWER than under the E1 control" in e3["note"]
    assert "stored references DEFLATE measured win rates under E3" in e3["note"]
    overall = row(df, "H1", "overall")
    assert overall["status"] == "supported"
    assert f"result driven by E3 {e3['estimate']:+.2f} pp vs E1 (stored references DEFLATE" in overall["note"]


def test_identical_stored_and_rerun_gives_no_effect():
    cube, trajs, plan = factorial_cube(identical=True), trajs_for(), AnalysisPlan()
    _, df = evaluate(cube, trajs, plan, B_policy=10)
    for env in ("E2", "E3", "E4"):
        r = h1_diff(df, env)
        assert r["estimate"] == 0 and r["ci_lo"] == 0 and r["ci_hi"] == 0
        assert r["status"] == "no effect detected"
    assert row(df, "H1", "overall")["status"] == "not supported"


def test_h1_without_pairs_is_inconclusive_and_notes_missing_pairs():
    cube, plan = factorial_cube(R=2, N=50), AnalysisPlan()  # R=2 < age 3
    trajs = trajs_for(R=2)
    _, df = evaluate(cube, trajs, plan, B_policy=5)
    assert (h1_diff(df, "E3")["status"]) == "no data"
    assert row(df, "H1", "overall")["status"] == "inconclusive"
    # missing cells: drop the t02 rerun of one seed's incumbent at round 4 -> E2/E4 pairs (j=4, i=1) skipped
    cube = factorial_cube(R=5, N=50)
    trajs = trajs_for(R=5)
    s, d, k, r = cube._idx(0, "t02", trajs[0].inc_slot[1], ("round", 4))
    cube.correct[s, d, k, r] = -1
    _, df = evaluate(cube, trajs, plan, B_policy=5)
    e2 = h1_diff(df, "E2")
    assert "missing" in e2["note"] and e2["n"] == 3 * 3 - 1
    assert row(df, "H1", "E2", "descriptive")["n"] == 3 * 3 - 1
    assert row(df, "H1", "E3", "descriptive")["n"] == 3 * 3


def test_deterministic_given_plan_seed(drift_result):
    cube, trajs, plan, pr, df = drift_result
    _, again = evaluate(cube, trajs, plan)
    pd.testing.assert_frame_equal(df, again)
    _, other = evaluate(cube, trajs, plan, seed=plan.bootstrap.seed + 1)
    assert not np.allclose(
        df["ci_lo"].to_numpy(dtype=float), other["ci_lo"].to_numpy(dtype=float), equal_nan=True
    )


# --------------------------------------------------------------------------- H2 / H3 (re-simulated bootstrap)


COUNTS = [10, 20, 15, 25, 18, 30, 22, 35, 28, 38, 30, 39]  # P1 (vs slot 0) accepts everything; half are false


def greedy_plan() -> AnalysisPlan:
    return AnalysisPlan(schedule={0: "E1"}, ablations={})


def test_h2_frozen_reference_has_more_false_accepts():
    cube, plan = prefix_cube(COUNTS), greedy_plan()
    trajs = trajs_for(R=len(COUNTS) - 1)
    _, df = evaluate(cube, trajs, plan, B=200, B_policy=60)
    p1 = row(df, "H2", "P1-P2")
    # P1: 11 accepts / seed, false at t = 2, 4, 6, 8, 10; P2 (greedy, refreshed): 6 accepts, 0 false (‡)
    assert p1["estimate"] == pytest.approx((5 / 11 - 0) * 100)
    assert p1["n"] == 3 * (11 + 6)
    assert p1["ci_lo"] > 0 and p1["status"] == "supported"
    assert "P2: 0/18 false accepts, 18 GT-coupled ‡" in p1["note"]
    assert "CI excludes 0: FAR(P1) > FAR(P2)" in p1["note"] and "P1 admitted MORE false accepts" in p1["note"]
    # adopt-on-promote removes prompt staleness: P1b equals P2 here, but every accept of both is GT-coupled
    # (greedy, reference = the incumbent's own GT cell), so 0 - 0 is a design property, not a measurement
    p1b = row(df, "H2", "P1b-P2")
    assert p1b["estimate"] == 0 and p1b["ci_lo"] == 0 and p1b["ci_hi"] == 0
    assert p1b["status"] == "inconclusive" and "‡ every accept of P1b and P2 is GT-coupled" in p1b["note"]
    assert "observed direction: none, FAR(P1b) = FAR(P2)" in p1b["note"]
    # H3: P3 never refreshes under an unchanged env (ratio 0); its FAR equals P2's only by construction
    ratio = row(df, "H3", "P3/P2")
    assert ratio["estimate"] == 0 and ratio["unit"] == "ratio" and ratio["status"] == "supported"
    far = row(df, "H3", "P3-P2")
    assert far["estimate"] == 0 and far["status"] == "inconclusive"
    assert "‡ every accept of P3 and P2 is GT-coupled" in far["note"]
    overall = row(df, "H3", "overall")
    assert overall["status"] == "inconclusive" and "‡" in overall["note"]


def test_h2_inconclusive_with_few_accepts():
    counts = [30, 10, 12, 11, 9, 13]  # nothing ever beats slot 0 or the incumbent
    cube, plan = prefix_cube(counts), greedy_plan()
    _, df = evaluate(cube, trajs_for(R=len(counts) - 1), plan, B=100, B_policy=20)
    for subject in ("P1-P2", "P1b-P2"):
        r = row(df, "H2", subject)
        assert r["status"] == "inconclusive" and f"fewer than {MIN_ACCEPTS} accepts" in r["note"]
        assert np.isnan(r["estimate"])
    assert row(df, "H3", "P3-P2")["status"] == "inconclusive"
    assert row(df, "H3", "overall")["status"] == "inconclusive"


def test_policy_bootstrap_under_split_half():
    cube = prefix_cube(COUNTS, N=60)
    plan = AnalysisPlan(schedule={0: "E1"}, ablations={}, gt={"mode": "split_half"})
    trajs = trajs_for(R=len(COUNTS) - 1)
    _, df = evaluate(cube, trajs, plan, B=100, B_policy=30)
    r = row(df, "H2", "P1-P2")
    assert np.isfinite(r["estimate"]) and r["unit"] == "pp of FAR"
    assert "B=30" in r["note"]


def test_missing_policies_are_simulated_and_b_zero_gives_nan_ci():
    cube, plan = prefix_cube(COUNTS), greedy_plan()
    trajs = trajs_for(R=len(COUNTS) - 1)
    _, df = evaluate(cube, trajs, plan, B=0, B_policy=0, runs=[])
    p1 = row(df, "H2", "P1-P2")
    assert p1["estimate"] == pytest.approx(5 / 11 * 100)
    # without a CI nothing can be concluded (it used to read "not supported")
    assert np.isnan(p1["ci_lo"]) and p1["status"] == "inconclusive" and "no bootstrap CI" in p1["note"]
    # H1 without a CI cannot rule an effect in or out (it used to read "no effect detected" / "not supported")
    e3 = h1_diff(df, "E3")
    assert np.isfinite(e3["estimate"]) and e3["status"] == "no data" and "no bootstrap CI" in e3["note"]
    assert row(df, "H1", "overall")["status"] == "inconclusive"


def _stats(n_acc: int, n_false: int, ref_calls: float, rep_acc=None, rep_fa=None) -> _PolicyStats:
    return _PolicyStats([0, 1, 2], n_acc, n_false, 0, ref_calls, rep_acc, rep_fa)


@pytest.mark.parametrize(
    ("ref_p3", "fa_p3", "expected_far", "expected_overall"),
    [
        (400.0, 1, "supported", "supported"),  # cheaper and FAR within the margin
        (2200.0, 1, "supported", "not supported"),  # ratio >= 1
        (400.0, 15, "not supported", "not supported"),  # FAR difference entirely beyond +margin
        (400.0, 3, "inconclusive", "inconclusive"),  # CI [5, 15] straddles the +10 pp margin
    ],
)
def test_h3_status_rules(ref_p3, fa_p3, expected_far, expected_overall):
    B = 200
    rng = np.random.default_rng(0)
    acc = np.full(B, 20)
    # P2: FAR ~ 0.05; P3: FAR = fa_p3/20 with replicate jitter of +-1 false accept
    p2 = _stats(20, 1, 2200.0, acc, np.full(B, 1))
    p3 = _stats(20, fa_p3, ref_p3, acc, np.clip(fa_p3 + rng.integers(-1, 2, B), 0, 20))
    rows = pd.DataFrame(h3_rows({"P3": p3, "P2": p2}, AnalysisPlan(h3_equivalence_margin_pp=10.0), B, ""))
    assert list(rows["subject"]) == ["P3/P2", "P3-P2", "overall"]
    assert rows.iloc[0]["estimate"] == pytest.approx(ref_p3 / 2200.0)
    assert rows.iloc[0]["status"] == ("supported" if ref_p3 < 2200 else "not supported")
    assert rows.iloc[1]["status"] == expected_far
    assert rows.iloc[2]["status"] == expected_overall


def test_undefined_replicates_are_dropped_and_counted():
    a = _stats(10, 5, 0.0, np.array([10, 0, 8, 4]), np.array([5, 0, 4, 1]))
    b = _stats(10, 1, 0.0, np.array([10, 5, 0, 4]), np.array([1, 1, 0, 0]))
    est, lo, hi, n_ok, n_drop = _far_diff(a, b)
    assert est == pytest.approx(40.0) and (n_ok, n_drop) == (2, 2)
    assert lo == pytest.approx(np.quantile([40.0, 25.0], 0.025)) and hi == pytest.approx(
        np.quantile([40.0, 25.0], 0.975)
    )
    rows = pd.DataFrame(h2_rows({"P1": a, "P1b": a, "P2": b}, 4, ""))
    assert "2 of 4 replicates dropped (FAR undefined)" in rows.iloc[0]["note"]
    assert rows.iloc[0]["status"] == "supported"


# --------------------------------------------------------------------------- review regressions


def test_h1_partly_by_construction_pairs_are_flagged_with_counts():
    """Seed 0 gets physical greedy reruns with fresh outputs; seeds 1-2 keep cache-hit reruns. The control (E1)
    and the generation part of E3 are then 0 by construction in 2 of 3 seeds: that must be flagged (‡ k of n),
    not silently averaged into the control."""
    R, N = 6, 60
    cube = factorial_cube(R=R, N=N)
    rng = np.random.default_rng(rng_seed("test-hypotheses", "partial"))
    for k in range(R + 1):
        for r in range(k + 1, R + 1):
            v1 = rng.random(N) < 0.5
            set_cell(
                cube,
                0,
                "greedy",
                k,
                ("round", r),
                {"v1": v1, "v2": v1 | (rng.random(N) < 0.2)},
                physical=True,
            )
    trajs, plan = trajs_for(R=R), AnalysisPlan()
    _, df = evaluate(cube, trajs, plan, B=100, B_policy=5)
    n = 3 * (R - 3 + 1)
    k = 2 * (R - 3 + 1)
    raw_e1 = row(df, "H1", "E1", "descriptive")
    assert f"‡ {k} of {n} E1 pairs at age 3 are identical by construction" in raw_e1["note"]
    assert raw_e1["estimate"] != 0  # seed 0's physical reruns are measured
    raw_e3 = row(df, "H1", "E3", "descriptive")
    assert f"‡ generation part of E3 is 0 by construction in {k} of {n} pairs" in raw_e3["note"]
    e3 = h1_diff(df, "E3")
    assert f"‡ {k} of {n} E1 pairs" in e3["note"] and f"in {k} of {n} pairs" in e3["note"]
    # E2 (t02 reruns are independent draws) is never by construction
    assert "‡" not in row(df, "H1", "E2", "descriptive")["note"]


def test_h2_is_two_sided_and_states_the_direction():
    """Pre-registered H2 is a two-sided contrast: a CI entirely below 0 is an effect too (status per the plan),
    and the note says which policy admitted fewer false accepts."""
    B = 200
    acc = np.full(B, 20)
    p1 = _stats(20, 1, 0.0, acc, np.full(B, 1))  # FAR 5 %
    p2 = _stats(20, 10, 2200.0, acc, np.full(B, 10))  # FAR 50 %
    rows = pd.DataFrame(h2_rows({"P1": p1, "P1b": p1, "P2": p2}, B, ""))
    r = rows.iloc[0]
    assert r["estimate"] == pytest.approx(-45.0) and r["ci_hi"] < 0
    assert r["status"] == "supported"
    assert "FAR(P1) < FAR(P2)" in r["note"] and "FEWER false accepts" in r["note"]
    assert "two-sided; estimate -45.0 pp" in r["note"]
    same = pd.DataFrame(h2_rows({"P1": p2, "P1b": p2, "P2": p2}, B, "")).iloc[0]
    assert same["status"] == "not supported" and "CI includes 0" in same["note"]
    assert "observed direction: none, FAR(P1) = FAR(P2)" in same["note"]
    # a non-significant difference still states its observed direction
    p1_hi = _stats(20, 12, 0.0, acc, np.where(np.arange(B) % 2 == 0, 4, 16))  # CI straddles P2's 50 %
    p1_lo = _stats(20, 8, 0.0, acc, np.where(np.arange(B) % 2 == 0, 4, 16))
    up = pd.DataFrame(h2_rows({"P1": p1_hi, "P1b": p1_lo, "P2": p2}, B, ""))
    assert list(up["status"]) == ["not supported", "not supported"]
    assert (
        "observed direction: FAR(P1) > FAR(P2) by 10.0 pp (P1 admitted MORE false accepts"
        in up.iloc[0]["note"]
    )
    assert (
        "observed direction: FAR(P1b) < FAR(P2) by 10.0 pp (P1b admitted FEWER false accepts"
        in up.iloc[1]["note"]
    )


def _per_seed(
    seed: int, n_acc: int, n_false: int, ref_calls: float, B: int, coupled: int = 0
) -> _PolicyStats:
    return _PolicyStats([seed], n_acc, n_false, coupled, ref_calls, np.full(B, n_acc), np.full(B, n_false))


def test_contrasts_are_paired_over_the_seeds_both_policies_cover():
    """P2 lacks seed 2 (e.g. skipped for a missing cell): FAR(P1) - FAR(P2) and the call ratio must use seeds
    {0, 1} for BOTH policies, otherwise the contrast compares different seed populations."""
    B = 50
    p1 = _pool([_per_seed(0, 10, 2, 0, B), _per_seed(1, 10, 2, 0, B), _per_seed(2, 10, 10, 0, B)])
    p2 = _pool([_per_seed(0, 10, 1, 1100, B), _per_seed(1, 10, 1, 1100, B)])
    p3 = _pool([_per_seed(0, 10, 1, 200, B), _per_seed(1, 10, 1, 200, B), _per_seed(2, 10, 9, 9999, B)])
    assert p1.far == pytest.approx(14 / 30) and p1.restrict([0, 1]).far == pytest.approx(4 / 20)
    np.testing.assert_array_equal(p1.restrict([0, 1]).rep_fa, np.full(B, 4))
    h2 = pd.DataFrame(h2_rows({"P1": p1, "P1b": p1, "P2": p2}, B, "")).iloc[0]
    assert h2["estimate"] == pytest.approx((4 / 20 - 2 / 20) * 100)
    assert h2["n"] == 40 and "paired over the seeds both policies cover: [0, 1]" in h2["note"]
    h3 = pd.DataFrame(h3_rows({"P3": p3, "P2": p2}, AnalysisPlan(), B, ""))
    assert h3.iloc[0]["estimate"] == pytest.approx(200 / 1100)  # seed 2's 9999 calls are not in the ratio
    assert h3.iloc[1]["estimate"] == pytest.approx(0.0)


def test_paired_seeds_end_to_end():
    cube, plan = prefix_cube(COUNTS), greedy_plan()
    trajs = trajs_for(R=len(COUNTS) - 1)
    runs = simulate_all(cube, trajs, plan, ENVS, n_dev=200, policies=["P1", "P1b", "P2", "P3"])
    runs = [r for r in runs if not (r.policy.name == "P2" and r.seed == 2)]
    _, df = evaluate(cube, trajs, plan, B=100, B_policy=20, runs=runs)
    p1 = row(df, "H2", "P1-P2")
    assert p1["n"] == 2 * (11 + 6) and "paired over the seeds both policies cover: [0, 1]" in p1["note"]
    assert p1["estimate"] == pytest.approx(5 / 11 * 100) and np.isfinite(p1["ci_lo"])


def test_h3_needs_min_accepts_like_h2():
    """FAR equivalence from a handful of accepts is not evidence: inconclusive (it used to read 'supported')."""
    B = 100
    acc = np.full(B, MIN_ACCEPTS - 1)
    p2 = _stats(MIN_ACCEPTS - 1, 0, 2200.0, acc, np.zeros(B, dtype=int))
    p3 = _stats(MIN_ACCEPTS - 1, 0, 400.0, acc, np.zeros(B, dtype=int))
    rows = pd.DataFrame(h3_rows({"P3": p3, "P2": p2}, AnalysisPlan(), B, ""))
    assert rows.iloc[0]["status"] == "supported"  # the call ratio is deterministic
    assert rows.iloc[1]["ci_lo"] == 0 and rows.iloc[1]["ci_hi"] == 0
    assert (
        rows.iloc[1]["status"] == "inconclusive"
        and f"fewer than {MIN_ACCEPTS} accepts" in rows.iloc[1]["note"]
    )
    assert rows.iloc[2]["status"] == "inconclusive"


def test_memoised_environments_simulate_identically():
    tags = extractor_tags()
    memo = _memo_envs(ENVS)
    for eid, env in ENVS.items():
        assert memo[eid] == memo[eid] and memo[eid].fingerprint(tags) == env.fingerprint(tags)
        assert (
            memo[eid].fingerprint() == env.fingerprint() and memo[eid].fingerprint(None) == env.fingerprint()
        )
        assert memo[eid].components(tags) == env.components(tags)
    cube, plan = factorial_cube(R=5, N=40), AnalysisPlan()
    traj = trajs_for(R=5)[1]
    items = np.random.default_rng(3).integers(0, 40, 40)
    for policy in ("P1", "P1b", "P2", "P3", "P5"):
        kw = dict(gt_mode=plan.gt.mode, extractor_tags=tags, items=items, gt_items=items)
        a = simulate(cube, traj, plan.env_schedule(), ENVS, policy, plan.promotion_rule, **kw)
        b = simulate(cube, traj, plan.env_schedule(), memo, policy, plan.promotion_rule, **kw)
        assert a.rounds == b.rounds and a.calls == b.calls and a.refreshes == b.refreshes, policy
