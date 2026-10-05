"""analyze(): every bundle frame from a small store built by tests/test_loader.build_run (no Pipeline)."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_loader import BuiltRun, build_run

from driftlab.analysis.bundle import EMPTY_REASONS, FRAME_COLUMNS, analyze, main_summary, policy_names
from driftlab.analysis.bundle_io import FRAME_NAMES, AnalysisBundle, bundle_dir, has_bundle, load_bundle
from driftlab.analysis.loader import load_cube
from driftlab.config import dump_yaml, load_config, load_plan
from driftlab.store.store import Store

REQUIRED_META = (
    "run_id",
    "run_dir",
    "config_hash",
    "plan_hash",
    "plan_version",
    "synthetic",
    "created_at",
    "seeds",
    "R",
    "N",
    "D",
    "schedule",
    "extractor_tags",
    "engine_fp",
    "config",
    "plan",
    "provenance",
    "driftlab_version",
    "warnings",
    "skipped_pairs",
)
REQUIRED_COLUMNS = {
    "hypotheses": (
        "hypothesis",
        "contrast",
        "subject",
        "estimate",
        "ci_lo",
        "ci_hi",
        "unit",
        "n",
        "status",
        "note",
    ),
    "ledger_summary": (
        "stage",
        "purpose",
        "seed",
        "n_requested",
        "n_executed",
        "n_cache_hits",
        "prompt_tokens",
        "completion_tokens",
        "wall_s",
    ),
    "audit": ("decoding_id", "slot", "repeat", "pct_text_identical", "pct_correct_flip", "n"),
    "truncation": ("seed", "decoding", "slot", "draw_kind", "draw", "trunc_rate"),
    "trajectory": (
        "seed",
        "round",
        "incumbent_slot",
        "candidate_slot",
        "inc_dev_acc",
        "cand_dev_acc",
        "advanced",
        "n_attempts",
        "is_fallback",
        "origin",
        "prompt_hash",
        "prompt_text",
    ),
    "environments": (
        "env_id",
        "decoding",
        "temperature",
        "extractor",
        "extractor_tag",
        "fingerprint",
        "description",
        "changed_vs_storage",
    ),
    "accuracy": ("seed", "env", "slot", "round", "gt_acc"),
}


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> BuiltRun:
    return build_run(tmp_path_factory.mktemp("bundle"), R=4, N=12, physical_ages=(1, 3), audit_repeats=2)


@pytest.fixture(scope="module")
def bundle(built: BuiltRun) -> AnalysisBundle:
    return analyze(built.run_dir, B=200, log=lambda _m: None)


def test_every_frame_written_and_non_empty(bundle: AnalysisBundle, built: BuiltRun):
    assert set(bundle.frames) == set(FRAME_NAMES)
    d = bundle_dir(built.run_dir)
    for name in FRAME_NAMES:
        assert (d / f"{name}.csv").exists(), name
        df = bundle.frames[name]
        assert len(df) > 0, f"{name} is empty: {bundle.meta['empty_frames'].get(name)}"
        assert set(FRAME_COLUMNS[name]) <= set(df.columns), name
    assert bundle.meta["empty_frames"] == {}
    for name, cols in REQUIRED_COLUMNS.items():
        assert set(cols) <= set(bundle.frames[name].columns), name
    assert has_bundle(built.run_dir)


def test_no_producer_failed(bundle: AnalysisBundle):
    assert not [w for w in bundle.meta["warnings"] if "failed" in w], bundle.meta["warnings"]
    assert bundle.meta["skipped_pairs"] == 0


def test_meta(bundle: AnalysisBundle, built: BuiltRun):
    m = bundle.meta
    for key in REQUIRED_META:
        assert key in m, key
    plan = load_plan(built.run_dir / "plan.yaml")
    assert m["run_id"] == built.run_id and m["seeds"] == [0, 1]
    assert (m["R"], m["N"], m["D"]) == (4, 12, 12)
    assert m["plan_hash"] == plan.plan_hash() and m["plan_version"] == "prereg_v1"
    assert m["config_hash"] == built.cfg.config_hash()
    assert m["schedule"] == {"0": "E1", "4": "E2", "8": "E4"}
    assert set(m["extractor_tags"]) == {"v1", "v2"}
    assert m["synthetic"] is True and bundle.synthetic
    assert m["engine_fp"] and isinstance(m["config"], dict) and isinstance(m["plan"], dict)
    assert isinstance(m["warnings"], list)
    assert m["policies"] == policy_names(plan) and "FIXEDAGE_k1" in m["policies"]


def test_round_trip(bundle: AnalysisBundle, built: BuiltRun):
    back = load_bundle(built.run_dir)
    assert back.synthetic is True
    assert set(back.frames) == set(FRAME_NAMES)
    for key in REQUIRED_META:
        assert key in back.meta
    assert back.meta["schedule"] == bundle.meta["schedule"]
    for name in FRAME_NAMES:
        a, b = bundle.frames[name], back.frames[name]
        assert list(a.columns) == list(b.columns), name
        assert a.shape == b.shape, name
        for c in a.columns:
            x = pd.to_numeric(a[c], errors="coerce")
            if x.notna().any() and a[c].dtype != object and not pd.api.types.is_string_dtype(a[c]):
                np.testing.assert_allclose(
                    x.to_numpy(dtype=float),
                    pd.to_numeric(b[c], errors="coerce").to_numpy(dtype=float),
                    rtol=1e-9,
                    equal_nan=True,
                    err_msg=f"{name}.{c}",
                )
    index = json.loads((bundle_dir(built.run_dir) / "bundle.json").read_text())
    assert sorted(index["frames"]) == sorted(FRAME_NAMES)


def test_frame_contents(bundle: AnalysisBundle, built: BuiltRun):
    f = bundle.frames
    # pairs: one row per (seed, env, j, i) over the full triangle
    assert len(f["pairs"]) == 2 * 4 * sum(j + 1 for j in range(1, 5))
    assert set(f["table4"]["env"]) == {"E1", "E2", "E3", "E4"}
    t7 = f["table7"]
    assert list(t7["age"]) == [0, 0, 1, 1, 3, 3, 5, 5, 10, 10]
    assert (t7.loc[t7["age"] >= 5, "n_pairs"] == 0).all()  # ages beyond R: kept, empty
    # trajectory incl. round 0 with prompts and attempts
    tr = f["trajectory"]
    assert len(tr) == 2 * 5 and list(tr.loc[tr["seed"] == 0, "round"]) == [0, 1, 2, 3, 4]
    assert list(tr.loc[tr["seed"] == 1, "prompt_text"]) == built.prompts[1]
    assert list(tr.loc[tr["seed"] == 0, "n_attempts"]) == [0] + [built.n_attempts[0][t] for t in range(1, 5)]
    assert list(tr.loc[tr["seed"] == 0, "incumbent_slot"]) == built.inc_slot(0)
    # environments: Table 2 source
    env = f["environments"].set_index("env_id")
    assert env.loc["E1", "changed_vs_storage"] == ""
    assert env.loc["E2", "changed_vs_storage"] == "decoding"
    assert env.loc["E3", "changed_vs_storage"] == "extractor"
    assert env.loc["E4", "changed_vs_storage"] == "decoding,extractor"
    assert env["fingerprint"].nunique() == 4 and env.loc["E3", "extractor_tag"].startswith("v2@")
    # policies: headline schedule, every plan policy (+ FIXEDAGE for A4) on both seeds
    pol = set(f["policy_summary"]["policy"])
    assert {"P1", "P1b", "P2", "P3", "P4_k3", "P5", "ORACLE", "FIXEDAGE_k1"} <= pol
    assert len(f["policy_rounds"]) == len(pol) * 2 * 4
    assert list(f["ablations"]["ablation"]) == ["A1", "A2", "A3", "A4"]
    assert len(f["candidates"]) == 2 * 4
    # ledger: engine rows aggregated by (stage, purpose, seed)
    led = f["ledger_summary"]
    assert set(led["purpose"]) >= {"eval_matrix", "trajectory_dev"}
    assert int(led.loc[led["purpose"] == "eval_matrix", "n_requested"].sum()) > 0
    assert len(f["audit"]) == 4
    # truncation per present cell; accuracy for r >= slot under every env
    tru = f["truncation"]
    assert set(tru["draw_kind"]) == {"round", "gt", "audit"} and tru["trunc_rate"].between(0, 1).all()
    acc = f["accuracy"]
    assert (acc["round"] >= acc["slot"]).all() and set(acc["env"]) == {"E1", "E2", "E3", "E4"}
    hyp = f["hypotheses"]
    assert {"H1", "H2", "H3"} == set(hyp["hypothesis"])
    assert set(hyp["status"]) <= {
        "supported",
        "not supported",
        "inconclusive",
        "effect detected",
        "no effect detected",
        "no data",
        "descriptive",
    }


def test_analysis_results_written(bundle: AnalysisBundle, built: BuiltRun):
    with Store(built.db, read_only=True) as st:
        payload = st.get_analysis_result(bundle.plan_hash, "bundle_summary")
    assert payload is not None and payload["run_id"] == built.run_id and payload["synthetic"] is True
    assert {h["hypothesis"] for h in payload["hypotheses"]} == {"H1", "H2", "H3"}


def test_main_summary(bundle: AnalysisBundle):
    text = main_summary(bundle)
    for token in ("H1", "H2", "H3", "T3", "T7", "SYNTHETIC", "P1", "age  3"):
        assert token in text, token
    # also works on a reloaded (CSV-typed) bundle
    assert "H3" in main_summary(load_bundle(bundle.meta["run_dir"]))


def test_partial_run_does_not_crash(tmp_path: Path):
    b = build_run(tmp_path, R=4, N=6, rounds_done={0: [1, 2], 1: [1]}, matrix_rounds=2, audit_repeats=1)
    bundle = analyze(b.run_dir, B=50, store_results=False, schedule_random_n=3, log=lambda _m: None)
    assert set(bundle.frames) == set(FRAME_NAMES)
    assert any("partial trajectories" in w for w in bundle.meta["warnings"])
    assert bundle.meta["R_observed"] == {"0": 2, "1": 1}
    assert len(bundle.frames["pairs"]) > 0 and len(bundle.frames["trajectory"]) == 3 + 2
    assert set(bundle.frames["policy_summary"]["policy"]) >= {"P1", "P2", "P3"}
    for name, reason in bundle.meta["empty_frames"].items():
        assert name in FRAME_NAMES and reason
    assert not [w for w in bundle.meta["warnings"] if "failed" in w], bundle.meta["warnings"]
    back = load_bundle(b.run_dir)
    assert set(back.frames) == set(FRAME_NAMES)
    with Store(b.db, read_only=True) as st:  # store_results=False: nothing written
        assert st.get_analysis_result(bundle.plan_hash, "bundle_summary") is None


def test_unscored_run_yields_empty_derived_frames(tmp_path: Path):
    b = build_run(tmp_path, R=2, N=4, score=False, audit_repeats=1)
    bundle = analyze(b.run_dir, B=20, store_results=False, schedule_random_n=2, log=lambda _m: None)
    assert any("no score" in w for w in bundle.meta["warnings"])
    assert bundle.frames["pairs"].empty and "pairs" in bundle.meta["empty_frames"]
    assert bundle.meta["skipped_pairs"] > 0
    assert len(bundle.frames["trajectory"]) == 2 * 3 and len(bundle.frames["truncation"]) > 0
    assert set(EMPTY_REASONS) >= set(bundle.meta["empty_frames"])
    # the hypotheses frame survives policies that cannot be simulated (it used to fail as a whole)
    hyp = bundle.frames["hypotheses"]
    assert "hypotheses" not in bundle.meta["empty_frames"] and len(hyp)
    h = hyp.loc[hyp["subject"] == "overall"].set_index("hypothesis")
    assert h.loc["H1", "status"] == "inconclusive" and h.loc["H3", "status"] == "inconclusive"
    assert hyp.loc[hyp["hypothesis"] == "H2", "note"].str.contains("not simulated", regex=False).all()
    save_ok = load_bundle(b.run_dir)
    assert set(save_ok.frames) == set(FRAME_NAMES)


def test_short_run_has_no_failed_producers(tmp_path: Path):
    """R=1 < schedule_randomization.n_changes=2: the exploratory frame is empty with a reason, not 'failed';
    an H3 equivalence built on a single accept per policy is inconclusive, not supported; the H2/H3 policy
    bootstrap uses the (pre-registered / overridden) B, not a silent cap of 500 replicates."""
    b = build_run(tmp_path, R=1, N=4, audit_repeats=1)
    bundle = analyze(b.run_dir, B=600, store_results=False, log=lambda _m: None)
    assert bundle.meta["B_policy"] == bundle.meta["B"] == 600
    hyp = bundle.frames["hypotheses"]
    assert hyp.loc[hyp["hypothesis"] == "H2", "note"].str.contains("B=600", regex=False).all()
    assert bundle.frames["schedule_random"].empty
    assert "n_changes=2" in bundle.meta["empty_frames"]["schedule_random"]
    assert not [w for w in bundle.meta["warnings"] if "failed" in w], bundle.meta["warnings"]
    h3 = hyp.loc[hyp["hypothesis"] == "H3"].set_index("subject")["status"]
    assert h3["P3-P2"] == "inconclusive" and h3["overall"] != "supported"


def _relabel(b: BuiltRun, *, synthetic: int, overrides: list[str]) -> None:
    """Emulate a non-mock run dir: rewrite config.yaml and set the store's synthetic flag directly (the Store
    API can never clear it, by design)."""
    cfg = load_config(b.run_dir / "config.yaml", overrides=overrides)
    (b.run_dir / "config.yaml").write_text(dump_yaml(cfg))
    con = sqlite3.connect(b.db)
    try:
        con.execute("UPDATE runs SET synthetic = ?", (int(synthetic),))
        con.commit()
    finally:
        con.close()


def test_synthetic_flag_comes_from_the_store(tmp_path: Path):
    b = build_run(tmp_path, R=2, N=4, audit_repeats=1)
    quick = dict(B=20, store_results=False, B_policy=5, schedule_random_n=2, log=lambda _m: None)
    # a non-mock config cannot hide a synthetic store
    _relabel(b, synthetic=1, overrides=["backend.kind=hf"])
    syn = analyze(b.run_dir, **quick)
    assert syn.meta["B_policy"] == 5  # an explicit cap still applies
    assert syn.synthetic and syn.meta["synthetic_note"] and "SYNTHETIC" in main_summary(syn)
    assert load_bundle(b.run_dir).synthetic
    # a real (non-synthetic) store with a non-mock config is not labelled synthetic
    _relabel(b, synthetic=0, overrides=["backend.kind=hf", "data.eval.n=99"])
    real = analyze(b.run_dir, **quick)
    assert real.synthetic is False and real.meta["synthetic_note"] == ""
    assert "SYNTHETIC" not in main_summary(real) and load_bundle(b.run_dir).synthetic is False
    assert any("4 eval items but the config asks for 99" in w for w in real.meta["warnings"])
    assert any("config.yaml hashes to" in w for w in real.meta["warnings"])


def test_main_summary_tolerates_missing_and_nan_columns(bundle: AnalysisBundle):
    frames = dict(bundle.frames)
    frames["policy_summary"] = bundle.frames["policy_summary"].drop(columns=["n_false_accepts", "n_accepted"])
    t7 = bundle.frames["table7"].copy()
    t7["n_pairs"] = np.nan
    frames["table7"] = t7
    text = main_summary(AnalysisBundle(meta=bundle.meta, frames=frames))
    assert "(0/0)" in text and "T7" in text


def test_main_summary_marks_by_construction_values(bundle: AnalysisBundle):
    """A FAR made only of GT-coupled accepts is 0 by construction and must carry the ‡ in the CLI digest too."""
    frames = dict(bundle.frames)
    pol = bundle.frames["policy_summary"].copy()
    m = pol["policy"] == "P2"
    pol.loc[m, ["n_accepted", "n_accepted_gt_coupled", "n_false_accepts"]] = [3, 3, 0]
    pol.loc[m, "far_pooled"] = 0.0
    frames["policy_summary"] = pol
    hyp = bundle.frames["hypotheses"].copy()
    hyp["note"] = hyp["note"].astype(object)
    hyp.loc[hyp["subject"] == "P1-P2", "note"] = "‡ every accept of P1 and P2 is GT-coupled"
    frames["hypotheses"] = hyp
    text = main_summary(AnalysisBundle(meta=bundle.meta, frames=frames))
    assert "FAR 0.00‡ (0/3)" in text
    p1p2 = next(line for line in text.splitlines() if "FAR(P1) - FAR(P2)" in line)
    assert p1p2.endswith("‡") and "zero by construction" in text


@pytest.mark.slow
def test_demo_sized_analyze_is_fast(tmp_path: Path):
    """configs/demo_mock-sized data (3 seeds x 11 rounds x 200 items) analyzes in < 60 s with B = 2000."""
    b = build_run(
        tmp_path, seeds=(0, 1, 2), R=11, N=200, physical_ages=(1, 3, 5, 10), audit_repeats=2, flip_rate=0.004
    )
    t0 = time.perf_counter()
    cube = load_cube(b.db)
    assert time.perf_counter() - t0 < 15 and cube.correct.shape == (3, 2, 12, 15, 2, 200)
    t0 = time.perf_counter()
    bundle = analyze(b.run_dir, B=2000, log=lambda _m: None)
    assert time.perf_counter() - t0 < 60
    assert bundle.meta["B_policy"] == 2000  # the pre-registered B for H2/H3 as well
    assert bundle.meta["empty_frames"] == {}
    assert not [w for w in bundle.meta["warnings"] if "failed" in w]
