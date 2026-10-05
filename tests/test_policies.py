"""Tests for the dry-run reference-refresh policy simulation (hand-built cubes; no DB)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from driftlab.analysis.cube import MissingCell, empty_cube, make_trajectory, set_cell
from driftlab.analysis.metrics import mean_sd, paired, wilson
from driftlab.analysis.policies import (
    CALL_PURPOSES,
    CANDIDATE_COLUMNS,
    CONVENTIONS,
    NO_REF,
    PER_SEED_COLUMNS,
    REF_COLUMNS,
    ROUND_COLUMNS,
    SUMMARY_COLUMNS,
    CostConvention,
    PolicyRun,
    RoundDecision,
    as_environments,
    candidates_frame,
    decision_policies,
    get_convention,
    parse_policy,
    per_seed_frame,
    runs_to_frames,
    simulate,
    simulate_all,
    summarize_policies,
)
from driftlab.config import AnalysisPlan, PromotionRule
from driftlab.environments import EnvSchedule, build_environments
from driftlab.keys import rng_seed

ENVS = build_environments()
RULE = PromotionRule()
TEAMMATE_SCHEDULE = EnvSchedule({0: "E1", 4: "E2", 8: "E4"})


def _draws(R: int, G: int = 1) -> list[tuple[str, int]]:
    return [("round", r) for r in range(R + 1)] + [("gt", g) for g in range(G)]


def random_cube(
    R: int = 11,
    N: int = 200,
    seeds: tuple[int, ...] = (0,),
    tag: str = "base",
    physical_greedy: bool = False,
    flip: float = 0.15,
):
    """Valid full-triangle cube: greedy reruns share the creation generation (unless ``physical_greedy``),
    every t02 round is an independent draw, one ("gt", 0) draw per slot and decoding; v2 is a superset of v1."""
    cube = empty_cube("test", seeds, ["greedy", "t02"], ["v1", "v2"], R + 1, _draws(R), N)
    for s in seeds:
        rng = np.random.default_rng(rng_seed("test-policies", tag, s))
        quality = rng.uniform(0.3, 0.8, size=R + 1)

        def outputs(q: float, rng: np.random.Generator = rng) -> dict[str, np.ndarray]:
            v1 = rng.random(N) < q
            return {"v1": v1, "v2": v1 | (rng.random(N) < 0.2)}

        for k in range(R + 1):
            created = outputs(quality[k])
            set_cell(cube, s, "greedy", k, ("round", k), created, physical=True)
            rows = cube.gen_rows(s, "greedy", k, ("round", k)).copy()
            for r in range(k + 1, R + 1):
                if physical_greedy:
                    f = rng.random(N) < flip
                    w1 = created["v1"] ^ f
                    w2 = (created["v2"] ^ f) | w1
                    set_cell(cube, s, "greedy", k, ("round", r), {"v1": w1, "v2": w2}, physical=True)
                else:
                    set_cell(cube, s, "greedy", k, ("round", r), created, gen_rows=rows)
            for r in range(k, R + 1):
                set_cell(cube, s, "t02", k, ("round", r), outputs(quality[k]), physical=True)
            set_cell(cube, s, "t02", k, ("gt", 0), outputs(quality[k]), physical=True)
            set_cell(cube, s, "greedy", k, ("gt", 0), created, gen_rows=rows)
    return cube


def det_cube(correct_items: dict[int, set[int]], N: int, R: int, seed: int = 0):
    """Greedy-only cube whose slot k answers exactly ``correct_items[k]`` (same under v1/v2, every round)."""
    cube = empty_cube("det", [seed], ["greedy"], ["v1", "v2"], R + 1, _draws(R, 0), N)
    for k in range(R + 1):
        v = np.zeros(N, dtype=np.int8)
        v[sorted(correct_items.get(k, set()))] = 1
        set_cell(cube, seed, "greedy", k, ("round", k), {"v1": v, "v2": v}, physical=True)
        rows = cube.gen_rows(seed, "greedy", k, ("round", k)).copy()
        for r in range(k + 1, R + 1):
            set_cell(cube, seed, "greedy", k, ("round", r), {"v1": v, "v2": v}, gen_rows=rows)
    return cube


def traj(seed: int, R: int, advanced=None):
    adv = advanced if advanced is not None else [False] + [t % 2 == 1 for t in range(1, R + 1)]
    return make_trajectory(seed, [f"s{seed}-p{k}" for k in range(R + 1)], adv, [0.5] * (R + 1))


def run(cube, policy, schedule=TEAMMATE_SCHEDULE, seed=0, **kw):
    return simulate(cube, traj(seed, cube.R), schedule, ENVS, policy, RULE, **kw)


# --------------------------------------------------------------------------- parsing / conventions


def test_parse_policy_names_kinds_and_labels():
    p1, p1b, p2, p3, p5, orc = (parse_policy(n) for n in ("P1", "P1b", "P2", "P3", "P5", "ORACLE"))
    assert (p1.kind, p1.adopt_on_promote, p1.label, p1.refresh_rule) == (
        "frozen",
        False,
        "Frozen reference",
        "Never refresh",
    )
    assert (p1b.kind, p1b.adopt_on_promote, p1b.label) == ("frozen_adopt", True, "Frozen, adopt on promote")
    assert (p2.kind, p2.label, p2.refresh_rule) == ("per_batch", "Per-batch refresh", "Refresh every batch")
    assert (p3.kind, p3.label, p3.refresh_rule) == (
        "env_triggered",
        "Environment-triggered refresh",
        "Refresh after detected environment changes",
    )
    assert p5.kind == "component_aware" and p5.adopt_on_promote
    assert p5.label.startswith("Component-aware refresh")
    assert (orc.kind, orc.adopt_on_promote, orc.label) == ("oracle", False, "Ground-truth control")
    p4 = parse_policy("P4_k3")
    assert (p4.name, p4.kind, p4.k, p4.adopt_on_promote) == ("P4_k3", "age_triggered", 3, True)
    assert p4.label == "Age-triggered refresh (every 3 rounds)"
    fa = parse_policy("fixedage_k1")
    assert (fa.name, fa.kind, fa.k, fa.adopt_on_promote) == ("FIXEDAGE_k1", "fixed_age", 1, False)
    assert parse_policy("oracle") is orc and parse_policy(p2) is p2
    with pytest.raises(ValueError):
        parse_policy("P9")


def test_conventions():
    assert CONVENTIONS["teammate_v1"] == CostConvention("teammate_v1")
    full = CONVENTIONS["full"]
    assert full.count_candidate_dev and full.proposer == "attempts" and full.count_reference_init
    assert get_convention(None) is CONVENTIONS["teammate_v1"] and get_convention("full") is full
    with pytest.raises(ValueError):
        get_convention("nope")


# --------------------------------------------------------------------------- (1) pinned teammate costs


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_teammate_costs_are_pinned(seed):
    cube = random_cube(R=11, N=200, seeds=(seed,))
    runs = {p: run(cube, p, seed=seed, n_dev=200) for p in ("P1", "P1b", "P2", "P3", "P5", "ORACLE")}
    assert runs["P1"].total_calls == 4411
    assert runs["P1b"].total_calls == 4411
    assert runs["P2"].total_calls == 6611
    assert runs["P3"].total_calls == 4811
    assert runs["P5"].total_calls == 4611
    assert runs["ORACLE"].total_calls == 6611
    assert runs["P2"].refreshes == 11 and runs["P3"].refreshes == 2 and runs["P1"].refreshes == 0
    assert (runs["P5"].refreshes, runs["P5"].rescores) == (1, 1)
    p1 = runs["P1"]
    assert p1.candidate_generation_calls == 11 * 201
    assert p1.evaluation_calls == 11 * 200 and p1.reference_calls == 0
    assert runs["P3"].reference_calls == 400 and runs["ORACLE"].calls["oracle_gt"] == 2200
    for r in runs.values():
        assert set(r.calls) == set(CALL_PURPOSES)
        assert sum(sum(d.calls.values()) for d in r.rounds) == r.total_calls  # no init cost in teammate_v1
    p3_kinds = [d.refresh_kind for d in runs["P3"].rounds]
    assert [d.round for d in runs["P3"].rounds if d.refresh_kind == "refresh"] == [4, 8]
    assert p3_kinds.count("none") == 9
    p5 = {d.round: d.refresh_kind for d in runs["P5"].rounds}
    assert (p5[4], p5[8]) == ("refresh", "rescore")


def test_teammate_costs_through_simulate_all_and_summary():
    cube = random_cube(R=11, N=200, seeds=(0, 1, 2))
    trajs = {s: traj(s, 11) for s in (0, 1, 2)}
    runs = simulate_all(cube, trajs, AnalysisPlan(), ENVS, n_dev=200)
    assert len(runs) == 7 * 3
    assert [r.policy.name for r in runs[:3]] == ["P1"] * 3  # policy-major
    summ = summarize_policies(runs).set_index("policy")
    assert list(summarize_policies(runs).columns) == list(SUMMARY_COLUMNS)
    assert summ.loc["P1", "calls_total"] == 4411
    assert summ.loc["P2", "calls_total"] == 6611
    assert summ.loc["P3", "calls_total"] == 4811
    assert summ.loc["P5", "calls_total"] == 4611
    assert summ.loc["P2", "refreshes"] == 11 and summ.loc["P3", "refreshes"] == 2
    assert summ.loc["P5", "rescores"] == 1
    assert summ.loc["P3", "label"] == "Environment-triggered refresh"
    assert summ.loc["P1", "n_seeds"] == 3 and summ.loc["P1", "n_evaluated"] == 33


# --------------------------------------------------------------------------- (2) P1 false accept

# slot -> items answered correctly (N = 10, greedy, identical every round)
STALE = {0: {0, 1}, 1: set(range(6)), 2: set(range(4)), 3: set(range(7))}


def test_p1_false_accept_on_candidate_that_only_beats_the_starting_prompt():
    cube = det_cube(STALE, N=10, R=3)
    sched = EnvSchedule({0: "E1"})
    p1, p1b, p2 = (run(cube, p, sched) for p in ("P1", "P1b", "P2"))
    assert [d.accepted for d in p1.rounds] == [True, True, True]
    d2 = p1.rounds[1]
    assert (d2.ref_slot, d2.ref_round, d2.ref_age, d2.inc_before) == (0, 0, 2, 1)
    assert (d2.wins, d2.losses) == (2, 0)
    assert d2.gt_cand == pytest.approx(0.4) and d2.gt_inc == pytest.approx(0.6)
    assert d2.false_accept and p1.n_false_accepts == 1 and p1.far == pytest.approx(1 / 3)
    # adopting (P1b) or refreshing (P2) compares against the current incumbent and rejects candidate 2
    for r in (p1b, p2):
        assert [d.accepted for d in r.rounds] == [True, False, True]
        assert r.n_false_accepts == 0 and r.far == 0.0
    assert p1b.rounds[1].ref_slot == 1 and p1b.rounds[1].ref_source == "adopt"
    assert all(d.ref_slot == 0 and d.ref_round == 0 for d in p1.rounds)
    assert [s.source for s in p1.refs] == ["initial"]
    assert p1.final_inc == 3 and p1.final_acc_canonical == pytest.approx(0.7)


# --------------------------------------------------------------------------- (3) ORACLE


def test_oracle_far_is_zero_and_it_never_adopts():
    cube = random_cube(R=11, N=60, seeds=(0,), tag="oracle")
    orc = run(cube, "ORACLE", n_dev=60)
    assert orc.refs == []
    assert orc.n_accepted > 0 and orc.n_false_accepts == 0 and orc.far == 0.0
    for d in orc.rounds:
        assert d.accepted == (d.gt_cand > d.gt_inc + 1e-9)
        assert (d.ref_slot, d.ref_round, d.ref_age, d.ref_id, d.ref_env) == (NO_REF, NO_REF, NO_REF, "", "")
        assert d.refresh_kind == "none" and d.calls["oracle_gt"] == 60
    assert orc.calls["oracle_gt"] == 11 * 60 and orc.calls["reference_init"] == 0
    full = run(cube, "ORACLE", conv="full", n_dev=60)
    assert full.calls["reference_init"] == 0  # ORACLE keeps no reference


# --------------------------------------------------------------------------- (4) greedy P2 FAR = 0


@settings(max_examples=25, deadline=None, derandomize=True)
@given(
    tag=st.integers(0, 10_000),
    change=st.integers(1, 5),
    physical=st.booleans(),
    kind=st.sampled_from(["net_win", "win_rate", "mcnemar"]),
)
def test_p2_far_is_zero_by_construction_under_greedy(tag, change, physical, kind):
    """Greedy + fresh reference + a rule that needs wins > losses: accept implies GT improvement."""
    cube = random_cube(R=5, N=30, seeds=(0,), tag=f"prop{tag}", physical_greedy=physical)
    sched = EnvSchedule({0: "E1", change: "E3"})
    rule = PromotionRule(kind=kind, tau=0.02, alpha=0.5)
    r = simulate(cube, traj(0, 5), sched, ENVS, "P2", rule)
    assert r.n_false_accepts == 0
    for d in r.rounds:
        if d.accepted:
            assert d.gt_cand > d.gt_inc


def test_p1_is_not_far_free_on_the_stale_case_while_p2_is():
    cube = det_cube(STALE, N=10, R=3)
    sched = EnvSchedule({0: "E1"})
    assert run(cube, "P1", sched).n_false_accepts > 0
    assert run(cube, "P2", sched).n_false_accepts == 0


def test_sampling_decisions_use_independent_gt_draws():
    R, N = 1, 4
    cube = empty_cube("t", [0], ["greedy", "t02"], ["v1", "v2"], R + 1, _draws(R), N)
    cells = {
        (0, ("round", 0)): [0, 0, 0, 0],
        (0, ("round", 1)): [0, 0, 0, 0],
        (1, ("round", 1)): [1, 1, 0, 0],
        (0, ("gt", 0)): [1, 1, 1, 0],
        (1, ("gt", 0)): [1, 0, 0, 0],
    }
    for dec in ("greedy", "t02"):
        for (k, draw), v in cells.items():
            set_cell(cube, 0, dec, k, draw, {"v1": v, "v2": v})
    sched = EnvSchedule({0: "E2"})
    r = simulate(cube, traj(0, R), sched, ENVS, "P2", RULE, canonical_env="E2")
    d = r.rounds[0]
    assert d.accepted and (d.wins, d.losses) == (2, 0)
    assert (d.gt_cand, d.gt_inc) == (0.25, 0.75) and d.false_accept
    same = simulate(cube, traj(0, R), sched, ENVS, "P2", RULE, gt_mode="same_draw", canonical_env="E2")
    assert not same.rounds[0].false_accept


# --------------------------------------------------------------------------- (5) P5 rescore


def test_p5_rescore_is_free_and_updates_extractor_at_storage():
    cube = random_cube(R=8, N=50, seeds=(0,), tag="rescore")
    sched = EnvSchedule({0: "E1", 4: "E3"})
    r = run(cube, "P5", sched, n_dev=50)
    assert (r.rescores, r.refreshes) == (1, 0)
    assert r.calls["rescore"] == 0 and r.calls["reference_refresh"] == 0
    assert r.total_calls == 8 * (50 + 1 + 50)
    d4 = r.rounds[3]
    assert d4.round == 4 and d4.refresh_kind == "rescore"
    assert (d4.ref_env, d4.ref_extractor, d4.ref_decoding) == ("E3", "v2", "greedy")
    resc = next(s for s in r.refs if s.source == "rescore")
    parent = next(s for s in r.refs if s.ref_id == resc.parent_id)
    assert parent.retired_round == 4 and parent.extractor_at_storage == "v1"
    assert (resc.extractor_at_storage, resc.env_id, resc.decoding) == ("v2", "E3", "greedy")
    assert resc.created_round == parent.created_round and resc.slot == parent.slot
    assert resc.rescored_round == 4
    assert d4.ref_age == 4 - parent.created_round  # the stored text keeps its age
    # the comparison used the stored text re-scored with v2, not a regeneration
    stored_round = parent.created_round
    rescored = cube.vec(0, "greedy", parent.slot, ("round", stored_round), "v2")
    cand = cube.vec(0, "greedy", 4, ("round", 4), "v2")
    assert (d4.wins, d4.losses) == (paired(cand, rescored).wins, paired(cand, rescored).losses)
    assert resc.acc_at_storage == pytest.approx(rescored.mean())
    # a convention may charge re-scores; P3 regenerates instead
    priced = run(cube, "P5", sched, n_dev=50, conv=CostConvention("priced", rescore_cost=7))
    assert priced.calls["rescore"] == 7
    p3 = run(cube, "P3", sched, n_dev=50)
    assert (p3.refreshes, p3.rescores, p3.calls["reference_refresh"]) == (1, 0, 50)


def test_p5_regenerates_when_decoding_changes():
    cube = random_cube(R=6, N=40, seeds=(0,), tag="p5dec")
    r = run(cube, "P5", EnvSchedule({0: "E1", 3: "E4"}), n_dev=40)
    assert (r.refreshes, r.rescores) == (1, 0)
    assert r.rounds[2].refresh_kind == "refresh" and r.rounds[2].ref_age == 0


# --------------------------------------------------------------------------- (6) fixed age / age-triggered


def test_fixedage_k1_reference_is_always_one_round_old():
    cube = random_cube(R=11, N=40, seeds=(0,), tag="fixed")
    r = run(cube, "FIXEDAGE_k1", n_dev=40)
    assert [d.ref_age for d in r.rounds] == [1] * 11
    for prev, d in zip(r.rounds, r.rounds[1:], strict=False):
        assert d.ref_slot == prev.inc_before  # the incumbent when candidate t-1 was evaluated
        assert d.ref_env == TEAMMATE_SCHEDULE.env_at(d.round - 1)
        assert d.refresh_kind == "fixed_age"
    assert r.rounds[0].refresh_kind == "none" and r.rounds[0].ref_source == "initial"
    assert r.refreshes == 10 and r.calls["reference_refresh"] == 10 * 40
    assert not any(s.source == "adopt" for s in r.refs)


def test_fixedage_k0_matches_per_batch_refresh():
    cube = random_cube(R=11, N=40, seeds=(0,), tag="fixed0", physical_greedy=True)
    a, b = run(cube, "FIXEDAGE_k0"), run(cube, "P2")
    assert [d.accepted for d in a.rounds] == [d.accepted for d in b.rounds]
    assert [(d.wins, d.losses) for d in a.rounds] == [(d.wins, d.losses) for d in b.rounds]
    assert a.total_calls == b.total_calls


def test_age_triggered_refreshes_every_k_rounds_without_promotions():
    R = 11
    cube = det_cube({0: set(range(10))}, N=10, R=R)  # slot 0 perfect, every candidate fails
    r = run(cube, "P4_k3", EnvSchedule({0: "E1"}), n_dev=10)
    assert r.n_accepted == 0
    assert [d.round for d in r.rounds if d.refresh_kind == "refresh"] == [3, 6, 9]
    assert [d.ref_age for d in r.rounds] == [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    assert math.isnan(r.far)


def test_env_triggered_refreshes_equal_schedule_changes():
    cube = random_cube(R=11, N=40, seeds=(0,), tag="p3")
    sched = EnvSchedule({0: "E1", 2: "E3", 5: "E4", 9: "E2"})
    r = run(cube, "P3", sched)
    assert r.refreshes == 3
    assert [d.round for d in r.rounds if d.refresh_kind == "refresh"] == [2, 5, 9]
    assert not any(d.ref_env_stale for d in r.rounds)
    p1 = run(cube, "P1", sched)
    assert [d.ref_env_stale for d in p1.rounds] == [t >= 2 for t in range(1, 12)]


# --------------------------------------------------------------------------- (7) full convention


def test_full_convention_adds_candidate_dev_attempts_and_initial_reference():
    R, N, D = 11, 200, 200
    cube = random_cube(R=R, N=N, seeds=(0,))
    attempts = {t: 1 + t % 3 for t in range(1, R + 1)}
    base = run(cube, "P1", n_dev=D)
    full = run(cube, "P1", n_dev=D, conv="full", proposer_attempts=attempts)
    assert full.calls["candidate_dev"] == R * D
    assert full.calls["proposer"] == sum(attempts.values())
    assert full.calls["reference_init"] == N
    assert full.total_calls == base.total_calls + R * D + sum(a - 1 for a in attempts.values()) + N
    assert full.reference_calls == N
    assert full.candidate_generation_calls == R * D * 2 + sum(attempts.values())
    p2 = run(cube, "P2", n_dev=D, conv="full", proposer_attempts=attempts)
    assert p2.total_calls == full.total_calls + R * N
    plan = AnalysisPlan(cost_convention="full")
    runs = simulate_all(
        cube, {0: traj(0, R)}, plan, ENVS, D, proposer_attempts_by_seed={0: attempts}, policies=["P1"]
    )
    assert runs[0].total_calls == full.total_calls and runs[0].convention == "full"


# --------------------------------------------------------------------------- (8) summaries


def _fake_run(seed: int, policy: str, outcomes, acc: float, acc_final: float, total: int) -> PolicyRun:
    """``outcomes``: list of (accepted, false_accept) per round."""
    spec = parse_policy(policy)
    rounds = [
        RoundDecision(
            seed=seed,
            policy=spec.name,
            round=t,
            env_id="E1",
            ref_id="x",
            ref_slot=0,
            ref_round=0,
            ref_env="E1",
            ref_age=t,
            refresh_kind="none",
            wins=0,
            losses=0,
            ties=10,
            n=10,
            accepted=a,
            gt_cand=0.5,
            gt_inc=0.5,
            false_accept=f,
            inc_before=0,
            inc_after=0,
            cand_slot=t,
            calls={"candidate_eval": 10},
        )
        for t, (a, f) in enumerate(outcomes, start=1)
    ]
    calls = dict.fromkeys(CALL_PURPOSES, 0)
    calls["candidate_eval"] = total
    return PolicyRun(seed, spec, rounds, [], calls, 0, 0, 0, acc, acc_final)


def test_summarize_policies_wilson_and_per_seed_mean_sd():
    T, F = True, False
    runs = [
        _fake_run(0, "P1", [(T, T), (T, F), (T, F), (T, F), (F, F)], 0.70, 0.60, 100),
        _fake_run(1, "P1", [(T, T), (T, T), (F, F)], 0.72, 0.62, 100),
        _fake_run(2, "P1", [(F, F), (F, F)], 0.74, 0.64, 100),
        _fake_run(0, "P2", [(T, F)], 0.8, 0.7, 100),
        _fake_run(1, "P2", [(F, F)], 0.6, 0.5, 101),
    ]
    df = summarize_policies(runs)
    assert list(df["policy"]) == ["P1", "P2"]
    p1 = df.set_index("policy").loc["P1"]
    m, sd, n = mean_sd([0.25, 1.0])
    assert p1["far_seed_mean"] == pytest.approx(m) and p1["far_seed_sd"] == pytest.approx(sd)
    assert p1["n_seeds_far"] == n == 2
    assert (p1["n_accepted"], p1["n_false_accepts"], p1["n_evaluated"]) == (6, 3, 10)
    p, lo, hi = wilson(3, 6)
    assert (p1["far_pooled"], p1["far_lo"], p1["far_hi"]) == pytest.approx((p, lo, hi))
    assert p1["gt_acc_mean"] == pytest.approx(0.72)
    assert p1["gt_acc_sd"] == pytest.approx(np.std([0.70, 0.72, 0.74], ddof=1))
    assert p1["gt_acc_final_mean"] == pytest.approx(0.62)
    p2 = df.set_index("policy").loc["P2"]
    assert p2["far_seed_mean"] == 0.0 and p2["far_seed_sd"] == 0.0 and p2["n_seeds_far"] == 1
    # calls: int when constant within every policy, else float means
    assert df["calls_total"].dtype == np.float64
    assert list(df["calls_total"]) == [100.0, 100.5]
    assert df["refreshes"].dtype == np.int64
    const = summarize_policies(runs[:3])
    assert const["calls_total"].dtype == np.int64 and int(const["calls_total"].iloc[0]) == 100
    nan_only = summarize_policies(runs[2:3])
    assert math.isnan(nan_only["far_seed_mean"].iloc[0]) and math.isnan(nan_only["far_pooled"].iloc[0])
    assert list(summarize_policies([]).columns) == list(SUMMARY_COLUMNS)


def test_runs_to_frames_and_per_seed_frame():
    cube = random_cube(R=6, N=30, seeds=(0, 1), tag="frames")
    trajs = {s: traj(s, 6) for s in (0, 1)}
    runs = simulate_all(cube, trajs, AnalysisPlan(), ENVS, n_dev=30, schedule=EnvSchedule({0: "E1", 3: "E3"}))
    rounds, refs = runs_to_frames(runs)
    assert list(rounds.columns) == list(ROUND_COLUMNS) and list(refs.columns) == list(REF_COLUMNS)
    assert len(rounds) == len(runs) * 6
    assert len(refs) == sum(len(r.refs) for r in runs)
    tot = rounds.groupby(["policy", "seed"], sort=False)["calls_round_total"].sum()
    for r in runs:
        assert tot.loc[(r.policy.name, r.seed)] == r.total_calls
    orc = rounds[rounds["policy"] == "ORACLE"]
    assert (orc["ref_slot"] == NO_REF).all() and (orc["calls_oracle_gt"] == 30).all()
    assert set(refs["source"]) <= {"initial", "refresh", "adopt", "rescore", "fixed_age"}
    ps = per_seed_frame(runs)
    assert list(ps.columns) == list(PER_SEED_COLUMNS) and len(ps) == len(runs)
    empty_rounds, empty_refs = runs_to_frames([])
    assert empty_rounds.empty and list(empty_rounds.columns) == list(ROUND_COLUMNS)
    assert empty_refs.empty and list(empty_refs.columns) == list(REF_COLUMNS)


def test_split_half_mode_uses_disjoint_item_halves():
    cube = random_cube(R=4, N=40, seeds=(0,), tag="split")
    plan = AnalysisPlan.model_validate({"gt": {"mode": "split_half"}})
    runs = simulate_all(cube, {0: traj(0, 4)}, plan, ENVS, n_dev=40, policies=["P2"])
    d = runs[0].rounds[0]
    assert d.n == 20 and d.wins + d.losses + d.ties == 20
    env = ENVS[plan.env_schedule().env_at(1)]
    expected = cube.gt_vec(0, env.decoding.id, 1, 1, env.extractor)[20:].mean()
    assert d.gt_cand == pytest.approx(expected)
    assert runs[0].calls["candidate_eval"] == 4 * 40  # physical cost is still N per evaluation


def test_tuple_environments_match_environment_objects():
    cube = random_cube(R=5, N=30, seeds=(0,), tag="tuples")
    tup = {"E1": ("greedy", "v1"), "E2": ("t02", "v1"), "E3": ("greedy", "v2"), "E4": ("t02", "v2")}
    a = simulate(cube, traj(0, 5), TEAMMATE_SCHEDULE, tup, "P3", RULE)
    b = simulate(cube, traj(0, 5), TEAMMATE_SCHEDULE, ENVS, "P3", RULE)
    assert [d.accepted for d in a.rounds] == [d.accepted for d in b.rounds]
    assert a.refreshes == b.refreshes
    assert set(as_environments(tup)) == set(ENVS)
    with pytest.raises(ValueError):
        simulate(cube, traj(0, 5), EnvSchedule({0: "E9"}), ENVS, "P1", RULE)


def test_missing_cells_raise_or_skip():
    cube = random_cube(R=4, N=20, seeds=(0,), tag="missing")
    cube.correct[0, 0, 3, 3] = -1  # greedy slot 3, ("round", 3)
    trajs = {0: traj(0, 4)}
    with pytest.raises(KeyError):
        simulate_all(cube, trajs, AnalysisPlan(), ENVS, 20, policies=["P1"])
    with pytest.warns(UserWarning):
        out = simulate_all(cube, trajs, AnalysisPlan(), ENVS, 20, policies=["P1"], on_missing="skip")
    assert out == []


def test_short_cube_and_unscored_extractor_are_missing_cells():
    """A trajectory longer than the eval matrix raises MissingCell (not ValueError), and an extractor without
    score rows is skipped like any missing cell under on_missing='skip'."""
    cube = random_cube(R=3, N=20, seeds=(0,), tag="short")
    with pytest.raises(MissingCell):
        simulate(cube, traj(0, 5), TEAMMATE_SCHEDULE, ENVS, "P1", RULE)
    with pytest.warns(UserWarning):
        assert simulate_all(cube, {0: traj(0, 5)}, AnalysisPlan(), ENVS, 20, on_missing="skip") == []
    full = random_cube(R=4, N=20, seeds=(0,), tag="v1only")
    cube = empty_cube("test", (0,), ["greedy", "t02"], ["v1"], 5, full.draws, 20)  # nothing scored with v2
    for d in ("greedy", "t02"):
        for k in range(5):
            for draw in full.draws:
                if full.has(0, d, k, draw):
                    set_cell(cube, 0, d, k, draw, {"v1": full.vec(0, d, k, draw, "v1")}, physical=True)
    e3 = EnvSchedule({0: "E1", 2: "E3"})  # rounds 2..4 need v2 scores
    with pytest.raises(KeyError):
        simulate_all(cube, {0: traj(0, 4)}, AnalysisPlan(), ENVS, 20, policies=["P5"], schedule=e3)
    with pytest.warns(UserWarning):
        out = simulate_all(
            cube,
            {0: traj(0, 4)},
            AnalysisPlan(),
            ENVS,
            20,
            policies=["P1", "P5"],
            schedule=e3,
            on_missing="skip",
        )
    assert out == []  # every policy scores the round-2.. candidates with v2


# --------------------------------------------------------------------------- T5 candidates


def test_candidates_frame_rows_and_decisions():
    seeds = (0, 1)
    cube = random_cube(R=6, N=40, seeds=seeds, tag="cands")
    trajs = {s: traj(s, 6) for s in seeds}
    plan = AnalysisPlan()
    runs = simulate_all(cube, trajs, plan, ENVS, n_dev=40)
    df = candidates_frame(cube, trajs, plan, ENVS, runs)
    dec_cols = decision_policies(plan)
    assert dec_cols == ["P1", "P2", "P3", "ORACLE"]
    assert list(df.columns) == [*CANDIDATE_COLUMNS, *dec_cols]
    assert len(df) == 12 and df["candidate_id"].iloc[0] == "s0-r1"
    sched = plan.env_schedule()
    for row in df.itertuples(index=False):
        t, seed = int(row.round), int(row.seed)
        assert row.env == sched.env_at(t)
        assert row.incumbent_slot == trajs[seed].inc_slot[t]
        env = ENVS[row.env]
        assert row.candidate_acc == pytest.approx(cube.gt_acc(seed, env.decoding.id, t, t, env.extractor))
        assert row.delta_pp == pytest.approx((row.candidate_acc - row.incumbent_acc) * 100)
        expect = "improves" if row.delta_pp > 1e-7 else ("worse" if row.delta_pp < -1e-7 else "ties")
        assert row.gt_outcome == expect
        assert row.advanced == trajs[seed].advanced[t]
    lookup = {(r.policy.name, r.seed): r for r in runs}
    for p in dec_cols:
        for row in df.itertuples(index=False):
            d = lookup[(p, row.seed)].rounds[row.round - 1]
            assert getattr(row, p) == ("accept" if d.accepted else "reject")
    # decision policies missing from ``runs`` are simulated on the fly
    only_p1 = [r for r in runs if r.policy.name == "P1"]
    df2 = candidates_frame(cube, trajs, plan, ENVS, only_p1)
    pd.testing.assert_frame_equal(df, df2)


# --------------------------------------------------------------------------- review regressions


def test_split_half_requires_both_disjoint_item_sets():
    """A single explicit half under split_half used to measure GT on ALL items (re-coupling decision and GT)."""
    cube = random_cube(R=4, N=40, seeds=(0,), tag="split2")
    t, sched = traj(0, 4), EnvSchedule({0: "E1"})
    with pytest.raises(ValueError, match="both"):
        simulate(cube, t, sched, ENVS, "P2", RULE, gt_mode="split_half", items=np.arange(20))
    with pytest.raises(ValueError, match="both"):
        simulate(cube, t, sched, ENVS, "P2", RULE, gt_mode="split_half", gt_items=np.arange(20, 40))
    with pytest.raises(ValueError, match="disjoint"):
        simulate(
            cube,
            t,
            sched,
            ENVS,
            "P2",
            RULE,
            gt_mode="split_half",
            items=np.arange(25),
            gt_items=np.arange(20, 40),
        )
    explicit = simulate(
        cube,
        t,
        sched,
        ENVS,
        "P2",
        RULE,
        gt_mode="split_half",
        items=np.arange(20),
        gt_items=np.arange(20, 40),
    )
    implicit = simulate(cube, t, sched, ENVS, "P2", RULE, gt_mode="split_half")
    assert [(d.wins, d.losses, d.gt_cand) for d in explicit.rounds] == [
        (d.wins, d.losses, d.gt_cand) for d in implicit.rounds
    ]
    assert explicit.rounds[0].gt_cand == pytest.approx(cube.gt_vec(0, "greedy", 1, 1, "v1")[20:].mean())
    assert not any(d.gt_coupled for d in explicit.rounds)  # disjoint items: never coupled


def test_resampled_items_drive_decisions_and_gt():
    """The H2 bootstrap path: one resampled index vector (repeats allowed) for decisions and GT."""
    cube = random_cube(R=5, N=30, seeds=(0,), tag="boot")
    idx = np.random.default_rng(rng_seed("test-boot")).integers(0, 30, size=30)
    r = simulate(cube, traj(0, 5), EnvSchedule({0: "E1"}), ENVS, "P2", RULE, items=idx, gt_items=idx)
    for d in r.rounds:
        cand = cube.vec(0, "greedy", d.round, ("round", d.round), "v1")[idx]
        ref = cube.vec(0, "greedy", d.inc_before, ("round", d.round), "v1")[idx]
        p = paired(cand, ref)
        assert (d.wins, d.losses, d.n) == (p.wins, p.losses, 30)
        assert d.gt_cand == pytest.approx(cand.mean()) and d.gt_inc == pytest.approx(ref.mean())
        assert d.gt_coupled
    assert r.calls["candidate_eval"] == 5 * 30  # costs count the full N, not the resample


def test_gt_coupled_flags_rounds_whose_decision_is_the_gt_comparison():
    """Coupled rounds satisfy wins - losses == n * (gt_cand - gt_inc), so they can never be false accepts."""
    sched = TEAMMATE_SCHEDULE  # E1 (greedy) rounds 1-3, then t02 segments
    for physical in (False, True):
        cube = random_cube(R=11, N=60, seeds=(0,), tag="coupled", physical_greedy=physical)
        runs = {p: run(cube, p, sched, n_dev=60) for p in ("P1", "P1b", "P2", "P3", "P5", "P4_k3", "ORACLE")}
        assert all(d.gt_coupled for d in runs["ORACLE"].rounds)
        # P2's reference IS the incumbent's round-t GT cell under greedy; never under independent t02 draws
        assert [d.gt_coupled for d in runs["P2"].rounds] == [t < 4 for t in range(1, 12)]
        # P3 keeps the round-0 reference through the E1 segment: coupled only if reruns are cache hits
        # P3's E1-segment reference (initial or adopted) is the incumbent's earlier generation: coupled
        # exactly when greedy reruns are cache hits of it
        assert [d.gt_coupled for d in runs["P3"].rounds[:3]] == [not physical] * 3
        for name, r in runs.items():
            assert r.n_gt_coupled == sum(d.gt_coupled for d in r.rounds)
            for d in r.rounds:
                if d.gt_coupled:
                    assert not d.false_accept, name
                    if name != "ORACLE":
                        assert d.wins - d.losses == round(d.n * (d.gt_cand - d.gt_inc)), name
                        assert d.ref_slot == d.inc_before
                if d.env_id in ("E2", "E4") and name != "ORACLE":
                    assert not d.gt_coupled  # independent GT draws
    # same_draw couples P2 under sampling too
    cube = random_cube(R=4, N=40, seeds=(0,), tag="same")
    sd = run(cube, "P2", EnvSchedule({0: "E2"}), gt_mode="same_draw", canonical_env="E2")
    assert all(d.gt_coupled for d in sd.rounds) and sd.n_false_accepts == 0
    # surfaced in the frames
    summ = summarize_policies(list(runs.values())).set_index("policy")
    assert summ.loc["ORACLE", "n_gt_coupled"] == 11
    assert summ.loc["P2", "n_gt_coupled"] == 3
    assert summ.loc["P2", "n_accepted_gt_coupled"] == sum(d.accepted for d in runs["P2"].rounds[:3])
    rounds_df, _ = runs_to_frames(list(runs.values()))
    assert rounds_df["gt_coupled"].dtype == bool
    assert int(rounds_df["gt_coupled"].sum()) == sum(r.n_gt_coupled for r in runs.values())
    ps = per_seed_frame(list(runs.values()))
    assert list(ps["n_gt_coupled"]) == [r.n_gt_coupled for r in runs.values()]


def test_gt_coupled_detects_cache_hit_references_in_greedy_segments():
    """With cached greedy reruns, P1b/P3 decisions equal P2's in an unchanged greedy segment (by construction)."""
    cube = random_cube(R=8, N=50, seeds=(0,), tag="cache")
    sched = EnvSchedule({0: "E1", 5: "E3"})
    p2, p3, p1b = (run(cube, p, sched) for p in ("P2", "P3", "P1b"))
    assert [d.accepted for d in p3.rounds] == [d.accepted for d in p2.rounds]
    assert all(d.gt_coupled for d in p3.rounds) and p3.n_false_accepts == 0
    # P1b keeps v1 scores after the extractor change: not coupled from round 5 on unless it adopted under v2
    for d in p1b.rounds:
        if d.round >= 5 and d.ref_extractor == "v1":
            assert not d.gt_coupled


def test_adoption_is_free_and_resets_reference_age():
    R, N = 5, 10
    cube = det_cube({k: set(range(k + 1)) for k in range(R + 1)}, N=N, R=R)  # every candidate improves
    sched = EnvSchedule({0: "E1"})
    p1, p1b, p4 = (run(cube, p, sched, n_dev=N) for p in ("P1", "P1b", "P4_k2"))
    for r in (p1, p1b, p4):
        assert r.n_accepted == R and r.n_false_accepts == 0
    assert p1b.total_calls == p1.total_calls and p1b.calls["reference_refresh"] == 0
    assert [d.ref_age for d in p1b.rounds] == [1] * R
    assert [d.ref_slot for d in p1b.rounds] == list(range(R))
    assert [s.source for s in p1b.refs] == ["initial"] + ["adopt"] * R
    assert [d.ref_age for d in p1.rounds] == list(range(1, R + 1))
    # P4 never reaches age 2 because every adoption resets the age
    assert p4.refreshes == 0 and p4.total_calls == p1.total_calls


def test_decisions_do_not_depend_on_the_cost_convention():
    cube = random_cube(R=11, N=60, seeds=(0,), tag="convs")
    attempts = {t: 3 for t in range(1, 12)}
    for p in ("P1", "P3", "P5", "FIXEDAGE_k1", "ORACLE"):
        a = run(cube, p, n_dev=60)
        b = run(cube, p, n_dev=60, conv="full", proposer_attempts=attempts)
        assert [d.accepted for d in a.rounds] == [d.accepted for d in b.rounds]
        assert b.total_calls > a.total_calls


def test_policy_spec_requires_k_for_age_based_kinds():
    from driftlab.analysis.policies import PolicySpec

    with pytest.raises(ValueError):
        PolicySpec("X", "age_triggered", True)
    with pytest.raises(ValueError):
        PolicySpec("Y", "fixed_age", False, -1)
    assert PolicySpec("Z", "age_triggered", True, 0).k == 0


def test_candidates_frame_ignores_runs_from_other_schedules():
    cube = random_cube(R=11, N=60, seeds=(0, 1), tag="cf-sched")
    trajs = {s: traj(s, 11) for s in (0, 1)}
    plan = AnalysisPlan()
    good = simulate_all(cube, trajs, plan, ENVS, 60)
    other = simulate_all(cube, trajs, plan, ENVS, 60, schedule=EnvSchedule({0: "E1", 2: "E4"}))
    expected = candidates_frame(cube, trajs, plan, ENVS, good)
    # ablation-style runs listed first must not leak into T5
    pd.testing.assert_frame_equal(candidates_frame(cube, trajs, plan, ENVS, other + good), expected)
    pd.testing.assert_frame_equal(candidates_frame(cube, trajs, plan, ENVS, other), expected)
    # a policy present for one seed only is completed by simulation instead of left blank
    partial = [r for r in good if not (r.policy.name == "P3" and r.seed == 1)]
    pd.testing.assert_frame_equal(candidates_frame(cube, trajs, plan, ENVS, partial), expected)
    assert set(expected["P3"]) <= {"accept", "reject"}


def test_gt_coupled_falls_back_to_matrix_semantics_without_gen_rows():
    """Without gen rows, a non-physical greedy rerun is a cache hit of the slot's creation generation."""
    stale = {0: {0, 1}, 1: set(range(6)), 2: set(range(4)), 3: set(range(7))}
    cube = det_cube(stale, N=10, R=3)
    cube.gen_row[:] = -1
    sched = EnvSchedule({0: "E1"})
    assert all(d.gt_coupled for d in run(cube, "P3", sched).rounds)  # reruns are non-physical cache hits
    cube.physical[:] = True  # every rerun now a fresh generation: only P2's same-cell reference is coupled
    assert not any(d.gt_coupled for d in run(cube, "P3", sched).rounds)
    assert all(d.gt_coupled for d in run(cube, "P2", sched).rounds)
