"""Crash / budget / restore resume: every interrupted path converges to the uninterrupted run's content digest."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from driftlab.config import REPO_ROOT, load_config
from driftlab.engine import FAIL_AFTER_ENV, InjectedFailure
from driftlab.pipeline import Pipeline
from driftlab.store import CellConflict, Store

SMOKE = REPO_ROOT / "configs" / "smoke_mock.yaml"
# small chunks so that crashes land inside matrix groups (chunk size never enters the content digest)
OVERRIDES = ["matrix.chunk_size.mock=50"]
UP_TO_AUDIT = ["data", "trajectory", "matrix", "score", "audit"]


def _quiet(_msg: str) -> None:
    return None


def _cfg():
    return load_config(SMOKE, OVERRIDES)


def _executing_chunks(store: Store, run_id: str, stage: str) -> int:
    return sum(1 for r in store.ledger_rows(run_id) if r["stage"] == stage and r["n_executed"] > 0)


def _ledger_executed(store: Store, run_id: str) -> int:
    return sum(r["n_executed"] for r in store.ledger_rows(run_id))


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Uninterrupted run: digest, executed count and executing chunks per stage."""
    with Pipeline(_cfg(), tmp_path_factory.mktemp("ref") / "run", log=_quiet) as p:
        res = p.run(UP_TO_AUDIT)
        assert res["status"] == "ok" and res["complete"]
        return {
            "digest": p.store.content_digest(p.run_id),
            "executed": _ledger_executed(p.store, p.run_id),
            "chunks": {s: _executing_chunks(p.store, p.run_id, s) for s in ("trajectory", "matrix", "audit")},
            "matrix_planned": res["progress"]["stages"]["matrix"]["planned"],
        }


def _finish(cfg, run_dir: Path, **kw) -> tuple[str, dict, int]:
    with Pipeline(cfg, run_dir, log=_quiet, **kw) as p:
        res = p.run(UP_TO_AUDIT)
        return p.store.content_digest(p.run_id), res, _ledger_executed(p.store, p.run_id)


def test_chunking_does_not_change_the_digest(reference: dict, tmp_path: Path) -> None:
    digest, res, _ = _finish(load_config(SMOKE), tmp_path / "default_chunks")
    assert res["complete"] and digest == reference["digest"]
    assert reference["chunks"]["matrix"] > 10  # the reference really is chunked finely


def test_crash_mid_matrix_env_var_then_resume(
    reference: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    crash_at = reference["chunks"]["trajectory"] + reference["chunks"]["matrix"] // 2
    monkeypatch.setenv(FAIL_AFTER_ENV, str(crash_at))
    p = Pipeline(_cfg(), run_dir, log=_quiet)
    with pytest.raises(InjectedFailure):
        p.run(UP_TO_AUDIT)
    st = p.status()
    assert st["stages"]["trajectory"]["complete"]
    assert 0 < st["stages"]["matrix"]["done"] < reference["matrix_planned"]  # crashed inside the matrix
    assert not st["complete"]
    p.close()
    monkeypatch.delenv(FAIL_AFTER_ENV)
    digest, res, executed = _finish(_cfg(), run_dir)
    assert res["status"] == "ok" and res["complete"]
    assert digest == reference["digest"]
    assert executed == reference["executed"]  # nothing committed was paid for twice


def test_crash_mid_trajectory_kwarg_then_resume(reference: dict, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    with Pipeline(_cfg(), run_dir, log=_quiet, fail_after_chunks=reference["chunks"]["trajectory"] // 2) as p:
        with pytest.raises(InjectedFailure):
            p.run(UP_TO_AUDIT)
        done = p.status()["stages"]["trajectory"]
        assert done["done"] < done["planned"] and not p.status()["complete"]
    digest, res, executed = _finish(_cfg(), run_dir)
    assert res["complete"] and digest == reference["digest"] and executed == reference["executed"]


def test_crash_during_audit_then_resume(reference: dict, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    with Pipeline(_cfg(), run_dir, log=_quiet) as p:
        p.run(["data", "trajectory", "matrix", "score"])
    with Pipeline(_cfg(), run_dir, log=_quiet, fail_after_chunks=1) as p:
        with pytest.raises(InjectedFailure):
            p.run(["audit"])
        assert not p.status()["stages"]["audit"]["complete"]
    digest, res, _ = _finish(_cfg(), run_dir)
    assert res["complete"] and digest == reference["digest"]


def test_changed_matrix_plan_raises_cell_conflict(reference: dict, tmp_path: Path) -> None:
    """Cells are first-write-wins: resuming a partial matrix under a CHANGED plan (allow_config_change) fails
    before any generation instead of silently mixing two plans; an unrelated config change or a deliberate
    engine change still resumes."""
    run_dir = tmp_path / "run"
    crash = reference["chunks"]["trajectory"] + 3
    with Pipeline(_cfg(), run_dir, log=_quiet, fail_after_chunks=crash) as p, pytest.raises(InjectedFailure):
        p.run(UP_TO_AUDIT)
    unrelated = load_config(SMOKE, [*OVERRIDES, "audit.repeats=2"])  # does not touch matrix cells
    with Pipeline(unrelated, run_dir, log=_quiet, allow_config_change=True) as p:
        assert p.run(["matrix"], max_minutes=1e-9)["status"] == "budget_exhausted"  # resumes, no conflict
        partial = p.status()["stages"]["matrix"]["done"]
        assert 0 < partial < reference["matrix_planned"]
    for change in (
        "matrix.physical_greedy_reruns=none",  # stored physical reruns (nonce) would become cache hits
        "matrix.mode=lean",  # t02 sampling seeds change
        "trajectory.user_template='Question: {question}'",  # every request changes
    ):
        cfg = load_config(SMOKE, [*OVERRIDES, change])  # (the mock's fingerprint covers the user template)
        with Pipeline(cfg, run_dir, log=_quiet, allow_config_change=True, allow_engine_change=True) as p:
            n_ledger = len(p.store.ledger_rows(p.run_id))
            with pytest.raises(CellConflict, match="first-write-wins"):
                p.run(["matrix"])
            assert len(p.store.ledger_rows(p.run_id)) == n_ledger  # nothing was generated
    engine = load_config(SMOKE, [*OVERRIDES, "backend.mock.greedy_flip_rate=0.2"])  # new engine fingerprint
    with Pipeline(engine, run_dir, log=_quiet, allow_config_change=True, allow_engine_change=True) as p:
        res = p.run(["matrix"])
        assert res["status"] == "ok" and res["progress"]["stages"]["matrix"]["complete"]


def test_tiny_budget_then_complete(reference: dict, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    with Pipeline(_cfg(), run_dir, log=_quiet) as p:
        res = p.run(UP_TO_AUDIT, max_minutes=1e-9)
        assert res["status"] == "budget_exhausted" and res["stage"] == "trajectory"
        assert res["executed"] > 0 and not res["progress"]["complete"]
        assert {r["stage"]: r["status"] for r in p.store.get_stage_status(p.run_id)}["trajectory"] == (
            "budget_exhausted"
        )
        res = p.run(UP_TO_AUDIT)
        assert res["status"] == "ok" and res["complete"]
        assert p.store.content_digest(p.run_id) == reference["digest"]


def test_repeated_tiny_budgets_always_progress(reference: dict, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    total_chunks = sum(reference["chunks"].values())
    with Pipeline(_cfg(), run_dir, log=_quiet) as p:
        calls = 0
        for _ in range(total_chunks + 5):
            res = p.run(UP_TO_AUDIT, max_minutes=1e-9)
            calls += 1
            if res["status"] == "ok":
                break
            assert res["executed"] > 0  # every budgeted call commits at least one generation chunk
        assert res["status"] == "ok" and res["complete"] and calls >= 3
        assert p.store.content_digest(p.run_id) == reference["digest"]
        assert _ledger_executed(p.store, p.run_id) == reference["executed"]


def test_restore_fresh_dir_from_snapshot_plus_shards(reference: dict, tmp_path: Path) -> None:
    src = tmp_path / "src"
    snap = tmp_path / "drive" / "snapshot.sqlite"
    state = {"groups": 0}
    holder: dict[str, Pipeline] = {}

    def hook(stage: str) -> None:  # Colab-style snapshot in the middle of the matrix
        if stage == "matrix":
            state["groups"] += 1
            if state["groups"] == 5:
                holder["p"].store.backup_to(snap)

    p = Pipeline(_cfg(), src, log=_quiet, checkpoint_hook=hook)
    holder["p"] = p
    res = p.run(UP_TO_AUDIT)
    assert res["complete"] and snap.is_file()
    src_digest = p.store.content_digest(p.run_id)
    src_ledger = len(p.store.ledger_rows(p.run_id))
    src_audit = p.store.get_audit_results(p.run_id)
    p.close()
    assert src_digest == reference["digest"]
    with Store(snap) as s:  # the snapshot really is a mid-matrix state
        assert s.count_rows("cells") > 0 and s.content_digest("smoke_mock") != src_digest

    restored = tmp_path / "restored"
    restored.mkdir()
    shutil.copy(snap, restored / "store.sqlite")
    shutil.copytree(src / "shards", restored / "shards")
    digest, res, _ = _finish(_cfg(), restored)
    assert res["status"] == "ok" and res["complete"]
    assert res["executed"] == 0  # everything newer than the snapshot came back from the shard log
    assert digest == reference["digest"]
    # audit_results are in the shard log too: the restored run skips the audit instead of re-submitting it
    assert res["stages"]["audit"]["skipped"] == "complete" and res["requested"] == 0
    with Store(restored / "store.sqlite") as s:
        assert s.get_audit_results("smoke_mock") == src_audit
        assert len(s.ledger_rows("smoke_mock")) == src_ledger

    # a shard dir elsewhere (e.g. Drive) works too
    restored2 = tmp_path / "restored2"
    restored2.mkdir()
    shutil.copy(snap, restored2 / "store.sqlite")
    digest2, res2, _ = _finish(_cfg(), restored2, shard_dir=src / "shards")
    assert res2["executed"] == 0 and digest2 == reference["digest"]

    # snapshot alone (no shards): the rest is regenerated deterministically
    lone = tmp_path / "lone"
    lone.mkdir()
    shutil.copy(snap, lone / "store.sqlite")
    digest3, res3, _ = _finish(_cfg(), lone)
    assert res3["executed"] > 0 and digest3 == reference["digest"]
