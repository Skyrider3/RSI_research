"""Tests for the ablations (T8) and the exploratory schedule randomization."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from driftlab.analysis.ablations import (
    ABLATION_COLUMNS,
    SCHEDULE_RANDOM_COLUMNS,
    format_schedule,
    randomize_schedules,
    run_ablations,
    schedule_flags,
    schedule_randomization_frame,
)
from driftlab.analysis.cube import empty_cube, make_trajectory, set_cell
from driftlab.analysis.policies import simulate_all
from driftlab.config import REPO_ROOT, AnalysisPlan, load_plan
from driftlab.environments import EnvSchedule, build_environments
from driftlab.keys import rng_seed

ENVS = build_environments()
ABLATIONS = {
    "A1": {"schedule": {0: "E1", 4: "E2", 8: "E4"}, "description": "Full system"},
    "A2": {"schedule": {0: "E1", 8: "E3"}, "description": "No decoding changes"},
    "A3": {"schedule": {0: "E1", 4: "E2"}, "description": "No extraction changes"},
    "A4": {"schedule": {0: "E1", 4: "E2", 8: "E4"}, "fixed_age": 1, "description": "Fixed reference age"},
}


def _plan(**kw) -> AnalysisPlan:
    return AnalysisPlan.model_validate({"ablations": ABLATIONS, **kw})


def random_cube(R: int = 11, N: int = 40, seeds: tuple[int, ...] = (0, 1), tag: str = "abl"):
    """Valid full-triangle cube (greedy reruns cached, independent t02 draws, one GT draw per slot)."""
    draws = [("round", r) for r in range(R + 1)] + [("gt", 0)]
    cube = empty_cube("test", seeds, ["greedy", "t02"], ["v1", "v2"], R + 1, draws, N)
    for s in seeds:
        rng = np.random.default_rng(rng_seed("test-ablations", tag, s))
        quality = rng.uniform(0.3, 0.8, size=R + 1)

        def outputs(q: float, rng: np.random.Generator = rng) -> dict[str, np.ndarray]:
            v1 = rng.random(N) < q
            return {"v1": v1, "v2": v1 | (rng.random(N) < 0.2)}

        for k in range(R + 1):
            created = outputs(quality[k])
            set_cell(cube, s, "greedy", k, ("round", k), created, physical=True)
            rows = cube.gen_rows(s, "greedy", k, ("round", k)).copy()
            for r in range(k + 1, R + 1):
                set_cell(cube, s, "greedy", k, ("round", r), created, gen_rows=rows)
            for r in range(k, R + 1):
                set_cell(cube, s, "t02", k, ("round", r), outputs(quality[k]), physical=True)
            set_cell(cube, s, "t02", k, ("gt", 0), outputs(quality[k]), physical=True)
    return cube


def trajs_for(seeds, R: int) -> dict:
    adv = [False] + [t % 3 == 1 for t in range(1, R + 1)]
    return {s: make_trajectory(s, [f"s{s}-p{k}" for k in range(R + 1)], adv) for s in seeds}


def pairs_frame() -> pd.DataFrame:
    """Minimal all-pairs frame: only the columns run_ablations relies on (+ seed, j, i)."""
    infl = {
        ("E1", 3): [0.0, 0.0],
        ("E2", 3): [0.01, 0.03],
        ("E3", 3): [0.02, 0.04],
        ("E4", 3): [0.05, 0.07],
        ("E1", 1): [0.0, 0.0],
        ("E2", 1): [0.10, 0.20],
        ("E3", 1): [0.5, 0.5],
        ("E4", 1): [0.30, 0.40],
    }
    rows = []
    for (env, age), vals in infl.items():
        for seed, v in enumerate(vals):
            rows.append({"seed": seed, "env": env, "j": 5, "i": 5 - age, "age": age, "inflation": v})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- (10) ablations


def test_run_ablations_rows_flags_and_inflation():
    R, N, D = 11, 40, 200
    seeds = (0, 1)
    cube, trajs, plan = random_cube(R, N, seeds), trajs_for(seeds, R), _plan()
    df = run_ablations(cube, trajs, plan, ENVS, pairs_frame(), n_dev=D)
    assert list(df.columns) == list(ABLATION_COLUMNS)
    assert list(df["ablation"]) == ["A1", "A2", "A3", "A4"]
    t = df.set_index("ablation")
    flags = t[["decoding_changes", "extraction_changes", "reference_age_variation"]]
    assert flags.loc["A1"].tolist() == [True, True, True]
    assert flags.loc["A2"].tolist() == [False, True, True]
    assert flags.loc["A3"].tolist() == [True, False, True]
    assert flags.loc["A4"].tolist() == [True, True, False]
    for c in ("decoding_changes", "extraction_changes", "reference_age_variation"):
        assert df[c].dtype == bool
    # inflation: age 3 (age 1 for A4) over the envs visited at/after the first change
    assert t.loc["A1", "drift_inflation_pp"] == pytest.approx(np.mean([0.01, 0.03, 0.05, 0.07]) * 100)
    assert t.loc["A2", "drift_inflation_pp"] == pytest.approx(3.0)
    assert t.loc["A3", "drift_inflation_pp"] == pytest.approx(2.0)
    assert t.loc["A4", "drift_inflation_pp"] == pytest.approx(np.mean([0.10, 0.20, 0.30, 0.40]) * 100)
    assert t.loc["A1", "inflation_envs"] == "E2,E4" and t.loc["A2", "inflation_envs"] == "E3"
    assert (t.loc["A1", "inflation_age"], t.loc["A4", "inflation_age"]) == (3, 1)
    assert t.loc["A1", "n_pairs"] == 4 and t.loc["A2", "n_pairs"] == 2
    assert t.loc["A1", "schedule"] == "0:E1, 4:E2, 8:E4"
    # policies: P3 slot is FIXEDAGE_k1 for A4; costs follow teammate_v1
    assert list(t["p3_policy"]) == ["P3", "P3", "P3", "FIXEDAGE_k1"]
    per_round = D + 1 + N
    assert t.loc["A1", "calls_P3"] == R * per_round + 2 * N
    assert t.loc["A2", "calls_P3"] == R * per_round + 1 * N
    assert t.loc["A3", "calls_P3"] == R * per_round + 1 * N
    assert t.loc["A4", "calls_P3"] == R * per_round + (R - 1) * N
    # values agree with a direct simulation
    runs = simulate_all(
        cube, trajs, plan, ENVS, D, policies=["P1", "P3"], schedule=EnvSchedule({0: "E1", 8: "E3"})
    )
    p3 = [r for r in runs if r.policy.name == "P3"]
    acc = sum(r.n_accepted for r in p3)
    fa = sum(r.n_false_accepts for r in p3)
    assert t.loc["A2", "n_accepted_P3"] == acc
    if acc:
        assert t.loc["A2", "far_P3"] == pytest.approx(fa / acc)
    assert t.loc["A2", "gt_acc_P3"] == pytest.approx(np.mean([r.final_acc_canonical for r in p3]))
    assert t.loc["A1", "far_P1"] == t.loc["A4", "far_P1"] or (
        math.isnan(t.loc["A1", "far_P1"]) and math.isnan(t.loc["A4", "far_P1"])
    )


def test_run_ablations_from_the_preregistered_plan():
    path = REPO_ROOT / "analysis_plans" / "prereg_v1.yaml"
    if not path.exists():
        pytest.skip("prereg plan not present")
    plan = load_plan(path)
    cube, trajs = random_cube(11, 30, (0,)), trajs_for((0,), 11)
    df = run_ablations(cube, trajs, plan, ENVS, pairs_frame(), n_dev=200)
    assert len(df) == 4 and set(df["ablation"]) == {"A1", "A2", "A3", "A4"}
    assert df.set_index("ablation").loc["A4", "reference_age_variation"] == False  # noqa: E712


def test_run_ablations_without_ablations_or_pairs():
    cube, trajs = random_cube(11, 20, (0,)), trajs_for((0,), 11)
    empty = run_ablations(cube, trajs, AnalysisPlan(), ENVS, pairs_frame(), n_dev=10)
    assert empty.empty and list(empty.columns) == list(ABLATION_COLUMNS)
    no_pairs = run_ablations(cube, trajs, _plan(), ENVS, pairs_frame().iloc[0:0], n_dev=10)
    assert len(no_pairs) == 4 and no_pairs["drift_inflation_pp"].isna().all()
    assert (no_pairs["n_pairs"] == 0).all()


def test_schedule_flags_and_format():
    f = schedule_flags(EnvSchedule({0: "E1", 4: "E2", 8: "E4"}), ENVS, 11)
    assert f == {
        "decoding_changes": True,
        "extraction_changes": True,
        "first_change": 4,
        "envs_after_change": ["E2", "E4"],
    }
    beyond = schedule_flags(EnvSchedule({0: "E1", 20: "E3"}), ENVS, 11)  # change never visited
    assert beyond["first_change"] is None and not beyond["extraction_changes"]
    assert beyond["envs_after_change"] == []
    assert format_schedule({8: "E3", 0: "E1"}) == "0:E1, 8:E3"


# --------------------------------------------------------------------------- (9) schedule randomization


def test_randomize_schedules_is_deterministic_and_valid():
    a = randomize_schedules(11, 50, 2, seed=7)
    assert a == randomize_schedules(11, 50, 2, seed=7)
    assert a != randomize_schedules(11, 50, 2, seed=8)
    assert randomize_schedules(11, 10, 2, seed=7) == a[:10]  # prefix-stable in n
    assert len(a) == 50
    for ch in a:
        assert ch[0] == "E1" and len(ch) == 3
        rounds = sorted(r for r in ch if r > 0)
        assert all(1 <= r <= 11 for r in rounds) and len(set(rounds)) == 2
        envs = [ch[r] for r in sorted(ch)]
        assert all(e in {"E2", "E3", "E4"} for e in envs[1:])
        assert all(x != y for x, y in zip(envs, envs[1:], strict=False))
        assert EnvSchedule(ch).change_rounds() == rounds
    assert len({tuple(sorted(c.items())) for c in a}) > 10
    assert randomize_schedules(3, 2, 3, seed=1)[0].keys() == {0, 1, 2, 3}
    assert randomize_schedules(5, 3, 0, seed=1) == [{0: "E1"}] * 3
    with pytest.raises(ValueError):
        randomize_schedules(3, 1, 4, seed=1)
    with pytest.raises(ValueError):
        randomize_schedules(5, 1, 2, seed=1, env_pool=("E2",), base="E2")


def test_schedule_randomization_frame_is_exploratory_and_consistent():
    R, N, D = 11, 30, 30
    seeds = (0, 1)
    cube, trajs = random_cube(R, N, seeds, tag="sr"), trajs_for(seeds, R)
    plan = _plan(schedule_randomization={"n": 4, "n_changes": 2, "seed": 3})
    df = schedule_randomization_frame(cube, trajs, plan, ENVS, n_dev=D)
    assert df.attrs.get("exploratory") is True
    assert list(df.columns) == list(SCHEDULE_RANDOM_COLUMNS)
    assert len(df) == 4 * len(seeds) * 4
    assert set(df["policy"]) == {"P1", "P2", "P3", "P5"}
    scheds = randomize_schedules(R, 4, 2, 3, env_pool=("E2", "E3", "E4"), base="E1")
    assert list(df.drop_duplicates("schedule_id")["schedule"]) == [format_schedule(s) for s in scheds]
    p = {k: g for k, g in df.groupby("policy")}
    assert (p["P1"]["refreshes"] == 0).all() and (p["P2"]["refreshes"] == R).all()
    assert (p["P3"]["refreshes"] == 2).all()
    assert ((p["P5"]["refreshes"] + p["P5"]["rescores"]) == 2).all()
    assert (p["P2"]["reference_calls"] == R * N).all()
    assert (p["P1"]["calls_total"] == R * (D + 1 + N)).all()
    small = schedule_randomization_frame(cube, trajs, plan, ENVS, n_dev=D, policies=("P1",), n=2)
    assert len(small) == 2 * len(seeds)
