"""Run-cost estimator: planner-rule counts, magnitudes per config, hours ordering, warnings, calibration."""

from __future__ import annotations

import itertools
import math

import pytest

from driftlab.config import REPO_ROOT, ExperimentConfig, load_config, load_plan
from driftlab.estimate import (
    BACKENDS,
    GPUS,
    THROUGHPUT,
    Estimate,
    ages_rerun_bounds,
    audit_slot_count,
    calibrate_from_ledger,
    count_requests,
    estimate,
    format_estimate,
    normalize_backend,
    normalize_gpu,
    physical_rerun_cells,
)

CONFIGS = REPO_ROOT / "configs"


def _load(name: str, overrides: list[str] | None = None) -> ExperimentConfig:
    return load_config(CONFIGS / f"{name}.yaml", overrides)


def _rows(counts: dict) -> dict[str, dict]:
    return {r["component"]: r for r in counts["rows"]}


# --------------------------------------------------------------------------- counts


def test_smoke_counts_follow_planner_rules() -> None:
    cfg = _load("smoke_mock")  # S=2, R=4, N=D=24, ages [0,1,3,5,10], full, gt 1, audit 1 x 24
    S, R, N, D, K = 2, 4, 24, 24, 5
    c = count_requests(cfg)
    rows = _rows(c)
    assert rows["slot-0 dev run"]["logical"] == S * D and rows["slot-0 dev run"]["executed"] == D
    assert rows["candidate dev runs"]["executed"] == S * R * D
    assert rows["proposer attempts"]["executed"] == math.ceil(S * R * 1.3) == 11
    assert rows["greedy creation cells"]["logical"] == S * K * N
    assert rows["greedy creation cells"]["executed"] == N + S * R * N
    assert rows["greedy rerun cells"]["logical"] == S * R * (R + 1) // 2 * N
    # exact planner rule over both reference modes, R=4, ages {0,1,3}: 8..10 cells per seed
    assert rows["greedy rerun cells"]["executed"] == S * 10 * N
    assert rows["greedy rerun cells"]["executed_min"] == S * 8 * N
    assert rows["t02 triangle"]["logical"] == rows["t02 triangle"]["executed"] == S * K * (K + 1) // 2 * N
    assert rows["t02 gt draws"]["executed"] == S * K * N
    assert rows["determinism audit"]["executed"] == 2 * 1 * 24 * 2
    assert c["executed_total"] == 1979 and c["executed_min_total"] == 1883 and c["logical_total"] == 2027
    assert c["executed_total"] == sum(r["executed"] for r in c["rows"])
    assert c["by_purpose"]["eval_matrix"]["executed"] == 216 + 480 + 720
    assert set(c["by_purpose"]) == {
        "trajectory_dev",
        "candidate_dev",
        "proposer",
        "eval_matrix",
        "gt_draw",
        "audit",
    }
    assert set(c["by_stage"]) == {"trajectory", "matrix", "audit"}
    assert c["proposer_max"] == S * R * 3


def test_full_config_magnitude() -> None:
    c = count_requests(_load("full"))
    # ~87k in the design notes counted chain-mode cells only (29/seed); the planner rule also makes the
    # incumbent-mode cells physical (36..54/seed), so the full run costs ~91k-102k executed requests.
    assert 90_000 <= c["executed_min_total"] < c["executed_total"] <= 102_000
    assert c["executed_total"] == 101_643 and c["executed_min_total"] == 90_843
    assert c["executed_total"] < c["logical_total"]
    assert c["params"]["rounds"] == 11 and c["params"]["seeds"] == 3
    rows = _rows(c)
    assert rows["greedy rerun cells"]["executed"] == 3 * 54 * 200
    assert rows["greedy rerun cells"]["executed_min"] == 3 * 36 * 200
    assert rows["t02 triangle"]["executed"] == 3 * 78 * 200


def test_lean_config_magnitude() -> None:
    c = count_requests(_load("lean"))
    assert 25_000 <= c["executed_total"] <= 32_000
    assert c["executed_min_total"] == c["executed_total"] == 28_443  # no trajectory-dependent rows
    assert c["executed_total"] < c["logical_total"]
    rows = _rows(c)
    assert rows["greedy rerun cells"]["executed"] == 0
    assert rows["t02 triangle"]["executed"] == 3 * 12 * 200
    assert rows["t02 triangle"]["logical"] == 3 * 78 * 200


def test_rerun_modes_ordering() -> None:
    none = count_requests(_load("full", ["matrix.physical_greedy_reruns=none"]))
    ages = count_requests(_load("full"))
    full_all = count_requests(_load("full", ["matrix.physical_greedy_reruns=all"]))
    assert none["executed_total"] < ages["executed_total"] < full_all["executed_total"]
    assert _rows(full_all)["greedy rerun cells"]["executed"] == 3 * 66 * 200
    assert 68_000 <= none["executed_total"] <= 70_000  # plan doc: full without physical reruns ~69k
    assert none["logical_total"] == ages["logical_total"] == full_all["logical_total"]


def test_ages_ignore_duplicates_and_out_of_range() -> None:
    a = count_requests(_load("full", ["matrix.physical_ages=[0,1,1,3,5,10,20,-1]"]))
    b = count_requests(_load("full"))
    assert a["executed_total"] == b["executed_total"]
    assert a["executed_min_total"] == b["executed_min_total"]


def test_age_zero_adds_incumbent_cells() -> None:
    """Age 0 makes (inc_slot[i], i) physical in incumbent mode (the 'fresh' / refresh cell). With age 1 also
    planned those cells are already covered (inc_slot[i] is inc_slot[i-1] or i-1), so compare without it."""
    with0 = count_requests(_load("full", ["matrix.physical_ages=[0,3,5,10]"]))
    without0 = count_requests(_load("full", ["matrix.physical_ages=[3,5,10]"]))
    assert without0["executed_total"] < with0["executed_total"]
    assert without0["executed_min_total"] < with0["executed_min_total"]
    assert ages_rerun_bounds(11, [0, 1, 3, 5, 10]) == ages_rerun_bounds(11, [1, 3, 5, 10])


# --------------------------------------------------------------------------- planner rule (ages)


def _rule_cells(R: int, ages: set[int], inc: list[int]) -> set[tuple[int, int]]:
    """Literal transcription of ARCHITECTURE section 4: cell (k, r), r > k, is physical if some pair
    (i, j=r), j in 1..R, i in 0..j, with inc_slot[i] == k or i == k has j - i in ages."""
    return {
        (k, r)
        for r in range(1, R + 1)
        for k in range(r)
        if any((inc[i] == k or i == k) and (r - i) in ages for i in range(r + 1))
    }


def _trajectories(R: int):
    """Every dev-gated inc_slot vector (any subset of accepted rounds 1..R-1)."""
    for bits in itertools.product([False, True], repeat=max(R - 1, 0)):
        inc, cur = [0, 0], 0
        for t in range(1, R):
            cur = t if bits[t - 1] else cur
            inc.append(cur)
        yield inc[: R + 1]


NEVER_11 = [0] * 12
ALWAYS_11 = [0, 0, *range(1, 11)]


def test_physical_rerun_cells_matches_architecture_rule() -> None:
    for R in (1, 2, 4, 7):
        for ages in ({0, 1, 3, 5, 10}, {1}, {0}, {2, 3}):
            for inc in _trajectories(R):
                assert physical_rerun_cells(R, ages, inc) == _rule_cells(R, ages, inc), (R, ages, inc)


def test_ages_bounds_match_brute_force() -> None:
    for R in range(1, 10):
        for ages in ([0, 1, 3, 5, 10], [1, 3, 5, 10], [0], [2, 4], [0, 1, 2, 3]):
            sizes = [len(physical_rerun_cells(R, ages, inc)) for inc in _trajectories(R)]
            assert ages_rerun_bounds(R, ages) == (min(sizes), max(sizes)), (R, ages)


def test_chain_only_formula_undercounts() -> None:
    """Regression: sum_{a>0} (R+1-a) (= 29 for the full config) is below every possible trajectory."""
    R, ages = 11, [0, 1, 3, 5, 10]
    chain_only = sum(R + 1 - a for a in ages if 0 < a <= R)
    assert chain_only == 29
    assert len(physical_rerun_cells(R, ages, NEVER_11)) == 36
    assert len(physical_rerun_cells(R, ages, ALWAYS_11)) == 54
    assert ages_rerun_bounds(R, ages) == (36, 54)
    assert chain_only < ages_rerun_bounds(R, ages)[0]


def test_inc_slots_make_counts_exact() -> None:
    cfg = _load("full")
    inc = {0: NEVER_11, 1: ALWAYS_11, 2: [0, 0, 0, 2, 2, 2, 5, 5, 5, 5, 9, 9]}
    c = count_requests(cfg, inc_slots=inc)
    want = sum(len(physical_rerun_cells(11, cfg.matrix.physical_ages, v)) for v in inc.values()) * 200
    row = _rows(c)["greedy rerun cells"]
    assert row["executed"] == row["executed_min"] == want == (36 + 54 + 43) * 200
    assert c["executed_total"] == c["executed_min_total"] and c["params"]["trajectory_known"]
    bounds = count_requests(cfg)
    assert bounds["executed_min_total"] <= c["executed_total"] <= bounds["executed_total"]
    with pytest.raises(ValueError, match="seed"):
        count_requests(cfg, inc_slots={0: NEVER_11})
    with pytest.raises(ValueError, match="R \\+ 1"):
        count_requests(cfg, inc_slots={s: [0] * 5 for s in (0, 1, 2)})
    with pytest.raises(ValueError, match="invalid inc_slot"):
        physical_rerun_cells(3, [1], [0, 1, 1, 2])  # inc_slot[1] must be 0


def test_fixed_trajectories_are_exact() -> None:
    static = count_requests(_load("full", ["trajectory.mode=static"]))
    assert static["executed_total"] == static["executed_min_total"]
    assert _rows(static)["greedy rerun cells"]["executed"] == 3 * 36 * 200
    always = count_requests(_load("full", ["trajectory.advance_rule=always"]))
    assert always["executed_total"] == always["executed_min_total"]
    assert _rows(always)["greedy rerun cells"]["executed"] == 3 * 54 * 200


def test_audit_and_gt_toggles() -> None:
    base = count_requests(_load("full"))
    no_audit = count_requests(_load("full", ["audit.enabled=false"]))
    assert base["executed_total"] - no_audit["executed_total"] == 1600
    gt2 = count_requests(_load("full", ["matrix.gt_draws=2"]))
    assert gt2["executed_total"] - base["executed_total"] == 3 * 12 * 200


def test_audit_slots_deduplicated_when_the_config_fixes_them() -> None:
    # static: the last incumbent is always slot 0, so the audit covers one slot, not two
    static = count_requests(_load("full", ["trajectory.mode=static"]))
    assert static["by_purpose"]["audit"]["executed"] == 1 * 2 * 200 * 2
    assert audit_slot_count(_load("full", ["trajectory.mode=static"])) == 1
    assert audit_slot_count(_load("full")) == 2
    assert audit_slot_count(_load("full", ["audit.slots=[first,'0',last,last_incumbent]"])) == 2
    assert audit_slot_count(_load("full", ["audit.slots=[first,'3']"])) == 2
    assert not any("rerun:{r}" in n for n in count_requests(_load("full"))["notes"])


def test_expected_attempts_clamped() -> None:
    cfg = _load("smoke_hf_cpu")  # max_attempts 1, one seed, one round, audit off
    c = count_requests(cfg, expected_attempts=2.5)
    assert _rows(c)["proposer attempts"]["executed"] == 1
    assert c["executed_total"] == c["logical_total"]  # single seed, R=1: nothing is shared
    assert count_requests(cfg, expected_attempts=0.2)["params"]["expected_attempts"] == 1.0


def test_plan_is_recorded() -> None:
    plan = load_plan(REPO_ROOT / "analysis_plans" / "prereg_v1.yaml")
    c = count_requests(_load("full"), plan)
    assert c["params"]["plan_hash"] == plan.plan_hash()
    assert c["params"]["reference_mode"] == "incumbent"


def test_default_config_matches_full() -> None:
    assert (
        count_requests(ExperimentConfig())["executed_total"]
        == count_requests(_load("full"))["executed_total"]
    )


# --------------------------------------------------------------------------- hours + warnings


def test_throughput_table_is_complete() -> None:
    for g in GPUS:
        for b in BACKENDS:
            lo, hi = THROUGHPUT[(b, g)]
            assert 0 < lo <= hi


def test_hours_ordering() -> None:
    cfg = _load("full")
    t4, a100 = estimate(cfg, "t4", "vllm"), estimate(cfg, "a100", "vllm")
    hf_t4, l4 = estimate(cfg, "t4", "hf"), estimate(cfg, "l4", "vllm")
    assert isinstance(t4, Estimate)
    assert t4.hours_lo < t4.hours_hi
    assert t4.hours_lo > a100.hours_lo and t4.hours_hi > a100.hours_hi
    assert hf_t4.hours_lo > t4.hours_lo
    assert a100.hours_hi < l4.hours_hi < t4.hours_hi
    assert 3.0 < t4.hours_lo < t4.hours_hi < 9.0  # plan doc: vLLM T4 full ~4-7 h (before the ages fix)
    assert t4.executed_total == count_requests(cfg)["executed_total"]
    assert t4.executed_min_total == count_requests(cfg)["executed_min_total"]
    # hours_lo uses the best-case count at high throughput, hours_hi the worst case at low throughput
    assert t4.hours_lo == pytest.approx(t4.executed_min_total * 300 / 2000 / 3600)
    assert t4.hours_hi == pytest.approx(t4.executed_total * 300 / 1000 / 3600)
    lean = estimate(_load("lean"), "t4", "vllm")
    assert lean.hours_lo < t4.hours_lo and lean.hours_hi < t4.hours_hi


def test_hours_scale_with_tokens() -> None:
    cfg = _load("full")
    a, b = estimate(cfg, avg_completion_tokens=300), estimate(cfg, avg_completion_tokens=600)
    assert b.hours_lo == pytest.approx(2 * a.hours_lo) and b.hours_hi == pytest.approx(2 * a.hours_hi)


def test_warnings() -> None:
    full, lean = _load("full"), _load("lean")
    hf = estimate(full, "t4", "hf")
    assert any("12 h" in w for w in hf.warnings)
    assert any("HF transformers on a T4" in w for w in hf.warnings)
    assert estimate(full, "a100", "vllm").warnings == []
    assert estimate(full, "t4", "vllm").warnings == []
    assert not any("HF transformers on a T4" in w for w in estimate(lean, "t4", "hf").warnings)
    assert any("SYNTHETIC" in w for w in estimate(full, "t4", "mock").warnings)
    assert any("smoke" in w for w in estimate(_load("smoke_hf_cpu"), "cpu", "hf").warnings)
    assert any("CPU" in w for w in estimate(_load("smoke_hf_cpu"), "cpu", "hf").warnings)


def test_normalization() -> None:
    assert normalize_gpu("T4") == "t4"
    assert normalize_gpu("A100-80GB") == "a100"
    assert normalize_gpu("NVIDIA L4") == "l4"
    # nvidia-smi names as reported on Colab
    assert normalize_gpu("Tesla T4") == "t4"
    assert normalize_gpu("NVIDIA A100-SXM4-40GB") == "a100"
    assert normalize_gpu("NVIDIA A100 80GB PCIe") == "a100"
    assert normalize_gpu("none") == "cpu"
    for other in ("NVIDIA A10G", "NVIDIA L40S", "NVIDIA H100 80GB HBM3", "Tesla V100-SXM2-16GB"):
        with pytest.raises(ValueError):
            normalize_gpu(other)
    assert normalize_backend("auto") == "vllm"
    assert normalize_backend("openai-compat") == "openai_compat"
    with pytest.raises(ValueError):
        normalize_gpu("h100")
    with pytest.raises(ValueError):
        normalize_gpu("l40s")
    with pytest.raises(ValueError):
        estimate(_load("full"), "t4", "tensorrt")


def test_format_estimate() -> None:
    e = estimate(_load("full"), "t4", "hf")
    text = format_estimate(e)
    assert "TOTAL" in text and f"{e.executed_total:,}" in text and f"{e.logical_total:,}" in text
    assert "planning assumption" in text and "Warnings:" in text
    lines = text.splitlines()
    table = [ln for ln in lines if ln.startswith(("trajectory ", "matrix ", "audit "))]
    assert len(table) == len(e.counts["rows"])
    # right-aligned numeric columns end at the same position
    assert len({len(ln) for ln in table}) == 1
    assert "Warnings" not in format_estimate(estimate(_load("full"), "a100", "vllm"))
    assert f"{e.executed_min_total:,} (best case)" in text
    assert "best case" not in format_estimate(estimate(_load("lean"), "a100", "vllm"))


def test_to_dict() -> None:
    d = estimate(_load("smoke_mock")).to_dict()
    assert d["executed_total"] == 1979 and d["gpu"] == "t4" and d["backend"] == "vllm"
    assert d["executed_min_total"] == 1883


# --------------------------------------------------------------------------- calibration


def test_calibrate_from_ledger() -> None:
    rows = [
        {"purpose": "eval_matrix", "n_executed": 100, "completion_tokens": 25_000, "wall_s": 20.0},
        {"purpose": "eval_matrix", "n_executed": 100, "completion_tokens": 35_000, "wall_s": 40.0},
        {"purpose": "eval_matrix", "n_executed": 0, "completion_tokens": 0, "wall_s": 0.5},  # cache hits
        {"purpose": "proposer", "n_executed": 3, "completion_tokens": None, "wall_s": 2.0},
    ]
    cal = calibrate_from_ledger(rows)
    assert cal["tokens_per_s"] == pytest.approx(1000.0)
    assert cal["mean_completion_tokens"] == pytest.approx(300.0)
    assert cal["n_rows"] == 2 and cal["n_executed"] == 200
    assert cal["mean_batch"] == pytest.approx(100.0)
    assert calibrate_from_ledger(rows, purposes=["eval_matrix"])["n_rows"] == 2
    with pytest.raises(ValueError):
        calibrate_from_ledger(rows, purposes=["audit"])
    with pytest.raises(ValueError):
        calibrate_from_ledger([])


def test_estimate_with_calibration() -> None:
    cfg = _load("full")
    cal = {"tokens_per_s": 1000.0, "mean_completion_tokens": 250.0}
    e = estimate(cfg, "t4", "hf", calibration=cal)
    assert e.calibrated and e.avg_completion_tokens == 250.0
    assert e.hours_lo == pytest.approx(e.executed_min_total * 250.0 / 1250.0 / 3600)
    assert e.hours_hi == pytest.approx(e.executed_total * 250.0 / 800.0 / 3600)
    assert "calibrated" in format_estimate(e)
    with pytest.raises(ValueError):
        estimate(cfg, calibration={"tokens_per_s": 0})
    with pytest.raises(ValueError):
        estimate(cfg, avg_completion_tokens=0)


def test_small_batch_calibration_warns_for_batched_engines() -> None:
    cfg = _load("full")
    small = {"tokens_per_s": 500.0, "mean_completion_tokens": 280.0, "mean_batch": 20.0}
    assert any("small chunks" in w for w in estimate(cfg, "a100", "vllm", calibration=small).warnings)
    assert not any("small chunks" in w for w in estimate(cfg, "a100", "hf", calibration=small).warnings)
    big = {**small, "mean_batch": 1024.0}
    assert not any("small chunks" in w for w in estimate(cfg, "a100", "vllm", calibration=big).warnings)
