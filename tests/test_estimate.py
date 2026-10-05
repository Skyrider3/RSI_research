"""Run-cost estimator: planner-rule counts, magnitudes per config, hours ordering, warnings, calibration."""

from __future__ import annotations

import math

import pytest

from driftlab.config import REPO_ROOT, ExperimentConfig, load_config, load_plan
from driftlab.estimate import (
    BACKENDS,
    GPUS,
    THROUGHPUT,
    Estimate,
    calibrate_from_ledger,
    count_requests,
    estimate,
    format_estimate,
    normalize_backend,
    normalize_gpu,
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
    assert rows["greedy rerun cells"]["executed"] == S * ((R + 1 - 1) + (R + 1 - 3)) * N  # ages 1, 3 <= R
    assert rows["t02 triangle"]["logical"] == rows["t02 triangle"]["executed"] == S * K * (K + 1) // 2 * N
    assert rows["t02 gt draws"]["executed"] == S * K * N
    assert rows["determinism audit"]["executed"] == 2 * 1 * 24 * 2
    assert c["executed_total"] == 1787 and c["logical_total"] == 2027
    assert c["executed_total"] == sum(r["executed"] for r in c["rows"])
    assert c["by_purpose"]["eval_matrix"]["executed"] == 216 + 288 + 720
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
    assert 85_000 <= c["executed_total"] <= 90_000
    assert c["executed_total"] < c["logical_total"]
    assert c["params"]["rounds"] == 11 and c["params"]["seeds"] == 3
    rows = _rows(c)
    assert rows["greedy rerun cells"]["executed"] == 3 * (11 + 9 + 7 + 2) * 200
    assert rows["t02 triangle"]["executed"] == 3 * 78 * 200


def test_lean_config_magnitude() -> None:
    c = count_requests(_load("lean"))
    assert 25_000 <= c["executed_total"] <= 32_000
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


def test_ages_ignore_zero_duplicates_and_out_of_range() -> None:
    a = count_requests(_load("full", ["matrix.physical_ages=[0,1,1,3,5,10,20]"]))
    b = count_requests(_load("full"))
    assert a["executed_total"] == b["executed_total"]


def test_audit_and_gt_toggles() -> None:
    base = count_requests(_load("full"))
    no_audit = count_requests(_load("full", ["audit.enabled=false"]))
    assert base["executed_total"] - no_audit["executed_total"] == 1600
    gt2 = count_requests(_load("full", ["matrix.gt_draws=2"]))
    assert gt2["executed_total"] - base["executed_total"] == 3 * 12 * 200


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
    assert 3.0 < t4.hours_lo < t4.hours_hi < 8.0  # plan doc: vLLM T4 full ~4-7 h
    assert t4.executed_total == count_requests(cfg)["executed_total"]
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


def test_to_dict() -> None:
    d = estimate(_load("smoke_mock")).to_dict()
    assert d["executed_total"] == 1787 and d["gpu"] == "t4" and d["backend"] == "vllm"


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
    tokens = e.executed_total * 250.0
    assert e.hours_lo == pytest.approx(tokens / 1250.0 / 3600)
    assert e.hours_hi == pytest.approx(tokens / 800.0 / 3600)
    assert "calibrated" in format_estimate(e)
    with pytest.raises(ValueError):
        estimate(cfg, calibration={"tokens_per_s": 0})
