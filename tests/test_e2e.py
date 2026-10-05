"""End-to-end integration test: the smoke_mock config through every pipeline stage (data -> trajectory ->
matrix -> score -> audit -> analyze + tables) into tmp_path, then cross-module checks on the result.

Everything runs once per module (about 15 s in total); nothing is written outside tmp_path.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from driftlab.analysis.bundle_io import FRAME_NAMES, SYNTHETIC_NOTE, load_bundle
from driftlab.cli import main
from driftlab.config import REPO_ROOT, load_config, load_plan
from driftlab.estimate import count_requests
from driftlab.pipeline import STAGES, Pipeline
from driftlab.reporting.render import ALL_TABLES
from driftlab.reporting.tables import PAPER_COLUMNS, REPORTED_PATH

REPRODUCED = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T7b", "T8", "T9", "T10")
HYPOTHESIS_STATUSES = {
    "supported",
    "not supported",
    "inconclusive",
    "effect detected",
    "no effect detected",
    "no data",
    "descriptive",
}


def _ledger_rows(db: Path) -> int:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        return int(con.execute("SELECT COUNT(*) FROM ledger").fetchone()[0])


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#", dtype=str, keep_default_na=False)


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    run_dir = tmp_path_factory.mktemp("e2e") / "smoke_mock"
    cfg = load_config(REPO_ROOT / "configs" / "smoke_mock.yaml")
    logs: list[str] = []
    with Pipeline(cfg, run_dir, log=logs.append) as p:
        result = p.run()
        status = p.status()
    tables = run_dir / "exports" / "tables"
    return SimpleNamespace(
        cfg=cfg,
        plan=load_plan(run_dir / "plan.yaml"),
        run_dir=run_dir,
        db=run_dir / "store.sqlite",
        tables=tables,
        result=result,
        status=status,
        logs=logs,
        bundle=load_bundle(run_dir),
        ledger_rows=_ledger_rows(run_dir / "store.sqlite"),
        shards=sorted(p.name for p in (run_dir / "shards").iterdir()),
        snapshot={tid: (tables / f"{tid}.md").read_text(encoding="utf-8") for tid in ("T3", "T4", "T7")},
    )


def test_every_stage_completes(run):
    res = run.result
    assert res["status"] == "ok" and res["complete"] is True and res["synthetic"] is True
    assert list(res["stages"]) == list(STAGES)
    assert all(run.status["stages"][s]["complete"] for s in STAGES)
    assert res["executed"] > 0 and res["executed"] <= res["requested"]
    assert not [m for m in run.logs if "WARNING" in m], [m for m in run.logs if "WARNING" in m]
    assert set(run.bundle.frames) == set(FRAME_NAMES)
    assert not run.bundle.meta["warnings"], run.bundle.meta["warnings"]
    assert not run.bundle.meta["empty_frames"], run.bundle.meta["empty_frames"]


def test_tables_have_the_paper_headers(run):
    assert (run.tables / ALL_TABLES).is_file()
    ids = list(REPRODUCED) + (["T3_reported", "T4_reported"] if REPORTED_PATH.is_file() else [])
    for tid in ids:
        for ext in ("md", "csv", "tex"):
            assert (run.tables / f"{tid}.{ext}").is_file(), f"{tid}.{ext}"
        header = "| " + " | ".join(PAPER_COLUMNS[tid]) + " |"
        md = (run.tables / f"{tid}.md").read_text(encoding="utf-8").splitlines()
        assert md[2] == header, tid
        assert list(_read_csv(run.tables / f"{tid}.csv").columns) == list(PAPER_COLUMNS[tid]), tid
    t7 = _read_csv(run.tables / "T7.csv")
    ages = [int(a) for a in run.plan.t7.ages]
    assert len(t7) == 2 * len(ages) == 10
    assert t7.iloc[:, 0].astype(int).tolist() == [a for a in ages for _ in range(2)]
    assert t7.iloc[0]["Drift inflation (pp)"].endswith("‡")  # E1 at age 0: stored == rerun cell


def test_t3_costs_follow_the_teammate_formula(run):
    cfg, plan = run.cfg, run.plan
    R, D, N, S = cfg.run.rounds, cfg.data.dev.n, cfg.data.eval.n, len(cfg.run.seeds)
    assert plan.cost_convention == "teammate_v1"
    rounds = run.bundle.frame("policy_rounds")
    refreshes = {
        p: (rounds.loc[rounds["policy"] == p, "refresh_kind"].astype(str) == "refresh").sum() / S
        for p in ("P1", "P2", "P3")
    }
    changes = [r for r in sorted(plan.schedule) if 0 < int(r) <= R]
    assert refreshes == {"P1": 0, "P2": R, "P3": len(changes)}
    expected = {p: R * (D + 1 + N) + int(k) * N for p, k in refreshes.items()}
    assert expected == {"P1": 196, "P2": 292, "P3": 220}  # R=4, N=D=24, one schedule change within R
    t3 = _read_csv(run.tables / "T3.csv")
    assert t3["Model calls ↓"].tolist() == [str(expected[p]) for p in ("P1", "P2", "P3")]
    assert t3["Reference refreshes ↓"].tolist() == [str(int(refreshes[p])) for p in ("P1", "P2", "P3")]
    summ = run.bundle.frame("policy_summary").set_index("policy")
    for p, calls in expected.items():
        assert int(summ.loc[p, "calls_total"]) == calls


def test_synthetic_marking_everywhere(run):
    assert run.bundle.synthetic and run.bundle.meta["synthetic_note"] == SYNTHETIC_NOTE
    assert SYNTHETIC_NOTE in (run.tables / ALL_TABLES).read_text(encoding="utf-8")
    for tid in REPRODUCED:
        assert SYNTHETIC_NOTE in (run.tables / f"{tid}.md").read_text(encoding="utf-8"), tid
        first = (run.tables / f"{tid}.csv").read_text(encoding="utf-8").splitlines()[0]
        assert first.startswith("# SYNTHETIC DATA"), tid
    with sqlite3.connect(f"file:{run.db}?mode=ro", uri=True) as con:
        assert con.execute("SELECT synthetic FROM runs").fetchone()[0] == 1


def test_ledger_matches_the_estimator(run):
    """Executed (cache-miss) counts per purpose equal the estimator's exact counts for the trajectories that
    Phase A produced; proposer calls equal the stored proposer attempts."""
    trajs = run.bundle.frame("trajectory")
    inc = {
        int(s): g.sort_values("round")["incumbent_slot"].astype(int).tolist()
        for s, g in trajs.groupby("seed")
    }
    est = count_requests(run.cfg, run.plan, inc_slots=inc)["by_purpose"]
    led = run.bundle.frame("ledger_summary").groupby("purpose")[["n_requested", "n_executed"]].sum()
    for purpose in ("trajectory_dev", "candidate_dev", "eval_matrix", "gt_draw", "audit"):
        assert int(led.loc[purpose, "n_executed"]) == est[purpose]["executed"], purpose
        assert int(led.loc[purpose, "n_requested"]) == est[purpose]["logical"], purpose
    with sqlite3.connect(f"file:{run.db}?mode=ro", uri=True) as con:
        n_prop = int(con.execute("SELECT COUNT(*) FROM proposals").fetchone()[0])
    assert int(led.loc["proposer", "n_executed"]) == n_prop


def test_scientific_invariants(run):
    b = run.bundle
    pairs = b.frame("pairs")
    np.testing.assert_allclose(
        pairs["inflation"], pairs["infl_extract"] + pairs["infl_generation"], atol=1e-12
    )
    np.testing.assert_allclose(
        pairs["inflation"], (pairs["w_stored"] - pairs["w_rerun"]) / pairs["n"], atol=1e-12
    )
    assert (pairs.loc[pairs["env"] == "E3", "inflation"] >= 0).all()  # v1-correct implies v2-correct
    e1_age0 = pairs.loc[(pairs["env"] == "E1") & (pairs["age"] == 0)]
    assert len(e1_age0) and e1_age0["by_construction"].astype(bool).all()
    hyp = b.frame("hypotheses")
    assert {"H1", "H2", "H3"} <= set(hyp["hypothesis"].astype(str))
    assert set(hyp["status"].astype(str)) <= HYPOTHESIS_STATUSES
    assert len(b.frame("audit")) > 0
    adv = b.frame("trajectory").query("round > 0")["advanced"].astype(str).str.lower().eq("true")
    assert 0 < adv.sum() < len(adv)  # the trajectory advances sometimes but not always
    with sqlite3.connect(f"file:{run.db}?mode=ro", uri=True) as con:
        evalq = [r[0][:80] for r in con.execute("SELECT question FROM items WHERE split = 'test'")]
        texts = [r[0] for r in con.execute("SELECT text FROM prompts")]
        texts += [
            r[0]
            for r in con.execute(
                "SELECT response FROM generations WHERE gen_key NOT IN (SELECT gen_key FROM cells)"
            )
        ]
    assert texts and not [t for t in texts if any(q in t for q in evalq)]  # no eval text reaches the proposer


def test_rerun_is_a_no_op(run):
    with Pipeline(run.cfg, run.run_dir, log=lambda _m: None) as p:
        again = p.run()
    assert again["status"] == "ok" and again["complete"] is True
    assert again["executed"] == 0 and again["requested"] == 0
    assert again["stages"]["matrix"]["cells_written"] == 0
    assert again["stages"]["audit"].get("skipped") == "complete"
    assert _ledger_rows(run.db) == run.ledger_rows
    assert sorted(p.name for p in (run.run_dir / "shards").iterdir()) == run.shards
    for tid, text in run.snapshot.items():  # deterministic analysis: identical tables
        assert (run.tables / f"{tid}.md").read_text(encoding="utf-8") == text, tid


def test_cli_status_and_verify_extractors(run, capsys):
    assert main(["status", "--run-dir", str(run.run_dir)]) == 0
    out = capsys.readouterr().out
    assert "complete: yes" in out and "[SYNTHETIC]" in out
    assert main(["verify-extractors"]) == 0
    assert "extractors frozen" in capsys.readouterr().out


def test_synthetic_run_under_results_dir_keeps_the_bundle(run, tmp_path):
    """write_tables refuses synthetic tables under a 'results' directory; the analyze stage logs it and is
    'partial' instead of failing after the bundle was built."""
    dst = tmp_path / "results" / "smoke_mock"
    shutil.copytree(run.run_dir, dst, ignore=shutil.ignore_patterns("tables"))
    logs: list[str] = []
    with Pipeline(run.cfg, dst, log=logs.append) as p:
        res = p.run(["analyze"])
        assert (
            p.store.query("SELECT status FROM stage_status WHERE stage = 'analyze'")[0]["status"] == "partial"
        )
    assert res["status"] == "ok" and "tables_skipped" in res["stages"]["analyze"]
    assert any("tables not written" in m for m in logs)
    assert (dst / "exports" / "analysis" / "bundle.json").is_file()
    assert not (dst / "exports" / "tables").exists()
