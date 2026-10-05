"""The ``driftlab`` CLI: argument handling, one-line errors and every subcommand (pipeline ones on smoke_mock)."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys

import pytest

from driftlab import cli
from driftlab.cli import EXIT_BUDGET, EXIT_ERROR, EXIT_OK, EXIT_USAGE, lock_path, main
from driftlab.config import REPO_ROOT, load_plan

SMOKE = REPO_ROOT / "configs" / "smoke_mock.yaml"
PLAN = REPO_ROOT / "analysis_plans" / "prereg_v1.yaml"


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    rc = main([str(a) for a in argv])
    out, err = capsys.readouterr()
    return rc, out, err


# --------------------------------------------------------------------------- parser


def test_help_lists_every_command(capsys):
    rc, out, _ = run(capsys, "--help")
    assert rc == EXIT_OK
    for name in (
        "estimate",
        "run",
        "status",
        "analyze",
        "tables",
        "demo",
        "extract",
        "audit",
        "freeze-plan",
        "verify-extractors",
        "dashboard",
        "info",
    ):
        assert name in out
    rc, out, _ = run(capsys, "run", "--help")
    assert rc == EXIT_OK and "--max-minutes" in out and "--allow-engine-change" in out
    rc, out, _ = run(capsys, "analyze", "--help")
    assert rc == EXIT_OK and "--allow-stale-scores" in out


def test_no_command_and_bad_usage(capsys):
    rc, out, _ = run(capsys)
    assert rc == EXIT_USAGE and "usage: driftlab" in out
    assert run(capsys, "frobnicate")[0] == EXIT_USAGE
    assert run(capsys, "estimate")[0] == EXIT_USAGE  # -c is required
    rc, out, _ = run(capsys, "--version")
    assert rc == EXIT_OK and out.startswith("driftlab ")


def test_help_is_fast_and_lazy():
    code = (
        "import sys, time; t = time.perf_counter(); from driftlab.cli import main; rc = main(['--help']); "
        "heavy = [m for m in ('torch', 'transformers', 'vllm', 'pandas', 'numpy', 'driftlab.pipeline') "
        "if m in sys.modules]; print('HEAVY', heavy, rc, round(time.perf_counter() - t, 2))"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True)
    line = [ln for ln in res.stdout.splitlines() if ln.startswith("HEAVY")][-1]
    assert line.startswith("HEAVY [] 0"), line


# --------------------------------------------------------------------------- info / estimate


def test_info(capsys):
    import driftlab

    rc, out, _ = run(capsys, "info", "--no-cuda")  # in-process: never imports torch
    assert rc == EXIT_OK and "CUDA:" not in out
    assert f"driftlab {driftlab.__version__}" in out and "optional packages" in out
    for pkg in ("torch", "transformers", "vllm", "streamlit"):
        assert pkg in out
    assert f"repo root: {REPO_ROOT}" in out
    # The full command (CUDA check imports torch) runs in a subprocess so this process stays torch-free.
    res = subprocess.run(
        [sys.executable, "-m", "driftlab.cli", "info"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert res.returncode == EXIT_OK, res.stderr
    assert res.stdout.startswith(f"driftlab {driftlab.__version__}")
    assert ("CUDA:" in res.stdout) == cli._importable("torch")


def test_estimate_full_config(capsys):
    rc, out, _ = run(capsys, "estimate", "-c", REPO_ROOT / "configs" / "full.yaml")
    assert rc == EXIT_OK
    assert "DriftLab estimate: full" in out and "TOTAL" in out and "Estimated wall time" in out
    rc, out, _ = run(
        capsys, "estimate", "-c", SMOKE, "--gpu", "a100", "--set", "run.rounds=2", "--tokens", "100"
    )
    assert rc == EXIT_OK and "R=2 rounds" in out and "mock" in out


def test_missing_config_is_a_one_line_error(capsys, tmp_path):
    rc, out, err = run(capsys, "estimate", "-c", tmp_path / "nope.yaml")
    assert rc == EXIT_ERROR and out == ""
    assert err.count("\n") == 1 and "not found" in err and "Traceback" not in err
    rc, _, err = run(capsys, "estimate", "-c", SMOKE, "--set", "run.rounds=notanint")
    assert rc == EXIT_ERROR and err.count("\n") == 1 and "Traceback" not in err


# --------------------------------------------------------------------------- extract / verify


def test_extract_all(capsys):
    rc, out, _ = run(capsys, "extract", "--extractor", "all", "\\boxed{1,234}")
    assert rc == EXIT_OK
    lines = out.strip().splitlines()
    assert len(lines) == 2 and lines[0].startswith("v1 (v1@") and lines[1].startswith("v2 (v2@")
    assert "value=1234" in lines[1] and "method=boxed" in lines[1] and "span=[7, 12)" in lines[1]
    assert "content='1,234'" in lines[1]
    assert "value=1234" not in lines[0]  # strict v1 rejects the thousands separator


def test_extract_reads_stdin(capsys, monkeypatch):
    import io

    monkeypatch.setattr(sys, "stdin", io.StringIO("so the answer is #### 42\n"))
    rc, out, _ = run(capsys, "extract", "--extractor", "v1", "-")
    assert rc == EXIT_OK and out.count("\n") == 1 and "value=42" in out and "method=hash" in out
    monkeypatch.setattr(sys, "stdin", io.StringIO("\\boxed{7}"))
    rc, out, _ = run(capsys, "extract", "--extractor", "v2")  # no TEXT: piped stdin is read
    assert rc == EXIT_OK and "value=7" in out


def test_extract_without_text_on_a_terminal_does_not_hang(capsys, monkeypatch):
    import io

    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

        def read(self, *a):  # pragma: no cover - reaching this would block a real terminal
            raise AssertionError("must not wait for terminal input")

    monkeypatch.setattr(sys, "stdin", Tty())
    rc, out, err = run(capsys, "extract")
    assert rc == EXIT_ERROR and out == "" and "TEXT" in err and err.count("\n") == 1


def test_verify_extractors(capsys, monkeypatch):
    from driftlab import extraction

    rc, out, _ = run(capsys, "verify-extractors")
    assert rc == (EXIT_ERROR if extraction.verify_frozen() else EXIT_OK)
    if rc == EXIT_OK:
        assert out.startswith("extractors frozen:")
    monkeypatch.setattr(extraction, "verify_frozen", lambda path=None: {"v1": ("aaaa", "bbbb")})
    rc, out, _ = run(capsys, "verify-extractors")
    assert rc == EXIT_ERROR and "v1: pinned aaaa, current bbbb" in out


# --------------------------------------------------------------------------- freeze-plan


def test_freeze_plan_writes_lock_is_idempotent_and_refuses_changes(capsys, tmp_path):
    plan = tmp_path / "prereg_v1.yaml"
    shutil.copy(PLAN, plan)
    rc, out, _ = run(capsys, "freeze-plan", "--plan", plan)
    assert rc == EXIT_OK
    lock = lock_path(plan)
    assert lock == tmp_path / "prereg_v1.lock.json" and lock.exists()
    rec = json.loads(lock.read_text())
    digest = load_plan(plan).plan_hash()
    assert rec["plan_hash"] == digest and f"plan_hash: {digest}" in out
    assert rec["file_sha256"] == hashlib.sha256(plan.read_bytes()).hexdigest()
    assert rec["frozen_at_utc"].endswith("+00:00") and "git_commit" in rec
    before = lock.read_bytes()

    rc, out, _ = run(capsys, "freeze-plan", "--plan", plan)  # same plan: nothing rewritten
    assert rc == EXIT_OK and "already frozen" in out and lock.read_bytes() == before

    plan.write_text(plan.read_text().replace("t4: {age: 3}", "t4: {age: 4}"))
    rc, _, err = run(capsys, "freeze-plan", "--plan", plan)
    assert rc == EXIT_ERROR and "refusing" in err and "--force" in err and lock.read_bytes() == before

    rc, out, _ = run(capsys, "freeze-plan", "--plan", plan, "--force")
    new = json.loads(lock.read_text())
    assert rc == EXIT_OK and new["plan_hash"] == load_plan(plan).plan_hash() != digest


def test_freeze_plan_refuses_comment_only_edits(capsys, tmp_path):
    plan = tmp_path / "p.yaml"
    shutil.copy(PLAN, plan)
    assert run(capsys, "freeze-plan", "--plan", plan)[0] == EXIT_OK
    plan.write_text(plan.read_text() + "\n# a late comment\n")
    rc, _, err = run(capsys, "freeze-plan", "--plan", plan)
    assert rc == EXIT_ERROR and "comments / formatting" in err
    assert run(capsys, "freeze-plan", "--plan", tmp_path / "missing.yaml")[0] == EXIT_ERROR


# --------------------------------------------------------------------------- dashboard


def test_dashboard_errors_and_launch(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_importable", lambda name: False)
    rc, _, err = run(capsys, "dashboard")
    assert rc == EXIT_ERROR and "streamlit is not installed" in err

    monkeypatch.setattr(cli, "_importable", lambda name: True)
    monkeypatch.setattr(cli, "_repo_root", lambda: tmp_path)
    rc, _, err = run(capsys, "dashboard")
    assert rc == EXIT_ERROR and "Home.py" in err

    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "Home.py").write_text("# home\n")
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    calls = []

    def fake_call(cmd, env=None, cwd=None):
        calls.append((cmd, env, cwd))
        return 0

    monkeypatch.setattr(subprocess, "call", fake_call)
    rc, out, _ = run(capsys, "dashboard", "--run-dir", run_dir, "--port", "8502")
    assert rc == EXIT_OK and "launching" in out
    cmd, env, cwd = calls[0]
    assert cmd[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert cmd[4] == str(tmp_path / "app" / "Home.py") and cmd[-2:] == ["--server.port", "8502"]
    assert env["DRIFTLAB_RUN"] == str(run_dir.resolve()) and cwd == str(tmp_path)
    assert run(capsys, "dashboard", "--run-dir", tmp_path / "nope")[0] == EXIT_ERROR


# --------------------------------------------------------------------------- pipeline-backed commands


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    pytest.importorskip("driftlab.pipeline")
    pytest.importorskip("driftlab.analysis.bundle")
    run_dir = tmp_path_factory.mktemp("cli") / "smoke"
    rc = main(["run", "-c", str(SMOKE), "--run-dir", str(run_dir), "-q"])
    return rc, run_dir


def test_run_all_stages_writes_tables(smoke_run):
    rc, run_dir = smoke_run
    assert rc == EXIT_OK
    tables = run_dir / "exports" / "tables"
    assert (tables / "all_tables.md").exists() and (tables / "T7.tex").exists()
    from driftlab.analysis.bundle_io import SYNTHETIC_NOTE

    assert SYNTHETIC_NOTE in (tables / "all_tables.md").read_text(encoding="utf-8")


def test_rerun_is_idempotent_and_status(smoke_run, capsys):
    _, run_dir = smoke_run
    rc, out, _ = run(capsys, "run", "-c", SMOKE, "--run-dir", run_dir, "--stages", "data,matrix,score", "-q")
    assert rc == EXIT_OK and "executed 0" in out and "SYNTHETIC" in out
    rc, out, _ = run(capsys, "status", "--run-dir", run_dir)
    assert rc == EXIT_OK
    assert "complete: yes" in out and "[SYNTHETIC]" in out
    for stage in ("data", "trajectory", "matrix", "score", "audit", "analyze"):
        assert any(line.startswith(stage) for line in out.splitlines()), stage


def test_status_is_read_only_and_never_builds_a_backend(smoke_run, capsys, monkeypatch, tmp_path):
    """`status` of a GPU run must not load the model: it pretty-prints the closed pipeline's read-only status."""
    import driftlab.pipeline as pipeline_mod

    _, run_dir = smoke_run
    copy = tmp_path / "gpu_run"
    shutil.copytree(run_dir, copy)
    cfg_text = (copy / "config.yaml").read_text()
    assert "kind: mock" in cfg_text
    (copy / "config.yaml").write_text(cfg_text.replace("kind: mock", "kind: vllm", 1))
    watched = ("config.yaml", "plan.yaml", "provenance.json", "store.sqlite")  # (-shm/-wal: any reader)
    before = {n: (copy / n).read_bytes() for n in watched if (copy / n).exists()}
    assert set(before) == set(watched)

    def no_backend(*a, **k):
        raise AssertionError("status must not build a backend")

    monkeypatch.setattr(pipeline_mod, "make_backend", no_backend)
    rc, out, err = run(capsys, "status", "--run-dir", copy)
    assert rc == EXIT_OK, err
    assert "complete: yes" in out and "[SYNTHETIC]" in out  # the store's synthetic flag, not the config
    matrix = [ln for ln in out.splitlines() if ln.startswith("matrix")]
    assert matrix and "1,680" in matrix[0] and matrix[0].split()[-1] == "yes"
    after = {n: (copy / n).read_bytes() for n in watched}
    assert after == before  # nothing in the run dir was rewritten


def test_corrupt_store_is_a_one_line_error(smoke_run, capsys, tmp_path):
    run_dir = tmp_path / "broken"
    run_dir.mkdir()
    shutil.copy(smoke_run[1] / "config.yaml", run_dir / "config.yaml")
    (run_dir / "store.sqlite").write_bytes(b"this is not a sqlite database" * 100)
    rc, out, err = run(capsys, "status", "--run-dir", run_dir)
    assert rc == EXIT_ERROR and out == ""
    assert err.count("\n") == 1 and "Traceback" not in err and "DatabaseError" in err


def test_config_change_is_a_one_line_error(smoke_run, capsys):
    _, run_dir = smoke_run
    rc, _, err = run(capsys, "run", "-c", SMOKE, "--run-dir", run_dir, "--set", "run.rounds=3", "-q")
    assert rc == EXIT_ERROR and "ConfigMismatch" in err and err.count("\n") == 1
    rc, _, err = run(capsys, "run", "-c", SMOKE, "--run-dir", run_dir, "--stages", "bogus", "-q")
    assert rc == EXIT_ERROR and "bogus" in err


def test_tables_command(smoke_run, capsys, tmp_path):
    _, run_dir = smoke_run
    out_dir = tmp_path / "tables"
    rc, out, _ = run(capsys, "tables", "--run-dir", run_dir, "--out", out_dir, "--extended")
    assert rc == EXIT_OK and f"tables: {out_dir / 'all_tables.md'}" in out
    assert (out_dir / "T3_extended.tex").exists() and (out_dir / "all_tables_extended.md").exists()
    rc, out, _ = run(capsys, "tables", "--run-dir", run_dir, "--out", tmp_path / "csv", "--formats", "csv")
    assert rc == EXIT_OK and "tables:" not in out
    assert {p.suffix for p in (tmp_path / "csv").iterdir()} == {".csv"}
    assert not any("_extended" in p.name for p in (tmp_path / "csv").iterdir())
    rc, _, err = run(capsys, "tables", "--run-dir", run_dir, "--out", tmp_path / "results" / "t")
    assert rc == EXIT_ERROR and "SYNTHETIC" in err and not (tmp_path / "results").exists()
    rc, _, err = run(capsys, "tables", "--run-dir", run_dir, "--formats", "pdf")
    assert rc == EXIT_ERROR and "pdf" in err


def test_tables_without_bundle(capsys, tmp_path):
    (tmp_path / "r").mkdir()
    rc, _, err = run(capsys, "tables", "--run-dir", tmp_path / "r")
    assert rc == EXIT_ERROR and "driftlab analyze" in err
    rc, _, err = run(capsys, "status", "--run-dir", tmp_path / "r")
    assert rc == EXIT_ERROR and "not a DriftLab run directory" in err


def test_analyze_and_audit_commands(smoke_run, capsys):
    _, run_dir = smoke_run
    rc, out, _ = run(capsys, "analyze", "--run-dir", run_dir, "--B", "50", "-q")
    assert rc == EXIT_OK and "DriftLab analysis: run smoke_mock" in out
    assert f"tables: {run_dir / 'exports' / 'tables' / 'all_tables.md'}" in out
    rc, out, _ = run(capsys, "audit", "--run-dir", run_dir, "-q")
    assert rc == EXIT_OK and "audit" in out


def test_analyze_refuses_stale_scores_and_the_suggested_command_fixes_them(smoke_run, capsys, tmp_path):
    import sqlite3

    run_dir = tmp_path / "stale"
    shutil.copytree(smoke_run[1], run_dir)
    con = sqlite3.connect(run_dir / "store.sqlite")
    try:
        con.execute("UPDATE scores SET ext_hash = 'feedfacecafe' WHERE extractor = 'v1'")
        con.commit()
    finally:
        con.close()
    rc, out, err = run(capsys, "analyze", "--run-dir", run_dir, "--B", "20", "-q")
    assert rc == EXIT_ERROR and out == "" and err.count("\n") == 1 and "Traceback" not in err
    cmd = f"driftlab run -c {run_dir / 'config.yaml'} --run-dir {run_dir} --stages score"
    assert "stale scores" in err and cmd in err and "--allow-stale-scores" in err

    rc, out, _ = run(capsys, "analyze", "--run-dir", run_dir, "--B", "20", "--allow-stale-scores", "-q")
    assert rc == EXIT_OK and "STALE SCORES (analysed with --allow-stale-scores): v1@feedfacecafe" in out
    t3 = (run_dir / "exports" / "tables" / "T3.md").read_text(encoding="utf-8")
    assert "1. STALE SCORES: computed with extractor hashes v1@feedfacecafe" in t3

    rc, out, _ = run(
        capsys, "run", "-c", run_dir / "config.yaml", "--run-dir", run_dir, "--stages", "score", "-q"
    )
    assert rc == EXIT_OK and "executed 0" in out
    rc, out, _ = run(capsys, "analyze", "--run-dir", run_dir, "--B", "20", "-q")
    assert rc == EXIT_OK and "STALE" not in out
    assert "STALE" not in (run_dir / "exports" / "tables" / "T3.md").read_text(encoding="utf-8")
    assert "pre-registration: " in out  # status depends on whether the repo plan has been frozen


def test_budget_exhausted_exit_code(capsys, tmp_path):
    pytest.importorskip("driftlab.pipeline")
    rc, out, _ = run(capsys, "run", "-c", SMOKE, "--run-dir", tmp_path / "b", "--max-minutes", "0", "-q")
    assert rc == EXIT_BUDGET and "budget_exhausted" in out and "re-run to continue" in out


def test_demo_quick(capsys, tmp_path):
    pytest.importorskip("driftlab.pipeline")
    pytest.importorskip("driftlab.analysis.bundle")
    run_dir = tmp_path / "demo"
    rc, out, _ = run(capsys, "demo", "--quick", "--run-dir", run_dir, "-q")
    assert rc == EXIT_OK
    assert "### Table 3. Main comparison of reference-refresh policies" in out
    assert "### Table 7. Effect of reference age" in out
    assert "DriftLab analysis: run smoke_mock" in out and "SYNTHETIC" in out
    assert (run_dir / "exports" / "analysis" / "bundle.json").exists()
    assert (run_dir / "exports" / "tables" / "all_tables.md").exists()
