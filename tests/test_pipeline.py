"""Pipeline orchestration: full smoke run, status, idempotent re-run, run-dir files, guards, analyze hand-off."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import yaml

from driftlab.config import REPO_ROOT, load_config
from driftlab.estimate import count_requests
from driftlab.pipeline import STAGES, Pipeline, run_experiment
from driftlab.planning import PURPOSE_AUDIT, PURPOSE_GT, PURPOSE_MATRIX, count_plan
from driftlab.store import ConfigMismatch, EngineMismatch, Store, list_shards

SMOKE = REPO_ROOT / "configs" / "smoke_mock.yaml"
UP_TO_AUDIT = ["data", "trajectory", "matrix", "score", "audit"]


def _quiet(_msg: str) -> None:
    return None


@pytest.fixture(scope="module")
def full_run(tmp_path_factory: pytest.TempPathFactory):
    run_dir = tmp_path_factory.mktemp("pipe") / "run"
    hooks: list[str] = []
    cfg = load_config(SMOKE)
    p = Pipeline(cfg, run_dir, log=_quiet, checkpoint_hook=hooks.append)
    res = p.run(UP_TO_AUDIT)
    yield p, res, hooks
    p.close()


def _ledger_executed(store: Store, run_id: str) -> int:
    return sum(r["n_executed"] for r in store.ledger_rows(run_id))


# --------------------------------------------------------------------------- full run


def test_full_smoke_run_completes(full_run) -> None:
    p, res, _ = full_run
    assert res["status"] == "ok" and res["complete"] and res["synthetic"]
    assert set(res["stages"]) == set(UP_TO_AUDIT)
    st = p.status()
    assert st["complete"] and st["synthetic"] and st["run_id"] == "smoke_mock"
    stages = st["stages"]
    assert set(stages) == set(STAGES)
    assert stages["data"] == {"planned": 48, "done": 48, "complete": True}
    assert stages["trajectory"]["done"] == stages["trajectory"]["planned"] == 8
    assert stages["trajectory"]["per_seed"] == {0: 4, 1: 4}
    assert stages["matrix"]["planned"] == stages["matrix"]["done"] == 2 * 15 * 24 * 2 + 2 * 5 * 24
    assert stages["score"]["unscored"] == 0 and stages["score"]["complete"]
    assert stages["audit"]["complete"] and stages["audit"]["planned"] == 4
    assert stages["analyze"]["complete"] is False  # analysis bundle not built: not part of "complete"
    assert st["executed"] == res["executed"] == _ledger_executed(p.store, p.run_id)
    row = p.store.get_run(p.run_id)
    assert row["synthetic"] == 1 and row["config_hash"] == p.cfg.config_hash()
    assert row["engine_fp"] == p.engine_fp and row["plan_hash"] == p.plan.plan_hash()


def test_run_dir_files(full_run) -> None:
    p, _, _ = full_run
    d = p.run_dir
    assert yaml.safe_load((d / "config.yaml").read_text()) == p.cfg.model_dump(mode="json")
    assert (d / "plan.yaml").read_bytes() == (REPO_ROOT / p.cfg.analysis_plan).read_bytes()
    assert (d / "provenance.json").is_file() and (d / "store.sqlite").is_file()
    assert list_shards(d / "shards")
    assert int(p.store.get_meta("last_shard_seq")) == max(seq for seq, _ in list_shards(d / "shards"))


def test_checkpoint_hook_called_per_stage_round_and_group(full_run) -> None:
    _, res, hooks = full_run
    assert hooks.count("trajectory") == 1 + 4 + 1  # round 0, rounds 1..4, end of stage
    assert hooks.count("matrix") == res["stages"]["matrix"]["groups_run"] + 1
    for s in ("data", "score", "audit"):
        assert hooks.count(s) == 1
    assert hooks.index("data") < hooks.index("trajectory") < hooks.index("matrix") < hooks.index("score")


def test_rerun_executes_nothing(full_run) -> None:
    p, _, _ = full_run
    executed, rows = _ledger_executed(p.store, p.run_id), len(p.store.ledger_rows(p.run_id))
    digest = p.store.content_digest(p.run_id)
    shards = len(list_shards(p.shard_dir))
    with Pipeline(p.cfg, p.run_dir, log=_quiet) as again:
        res = again.run(UP_TO_AUDIT)
        assert res["status"] == "ok" and res["complete"] and res["executed"] == 0 and res["requested"] == 0
        assert _ledger_executed(again.store, again.run_id) == executed
        assert len(again.store.ledger_rows(again.run_id)) == rows
        assert again.store.content_digest(again.run_id) == digest
    assert len(list_shards(p.shard_dir)) == shards  # nothing written at all


def test_ledger_matches_planner_counts(full_run) -> None:
    p, _, _ = full_run
    trajs = p.trajectories()
    by: dict[str, int] = {}
    for r in p.store.ledger_rows(p.run_id):
        by[r["purpose"]] = by.get(r["purpose"], 0) + r["n_executed"]
    c = count_plan(p.cfg, p.plan, trajs, p.eval_split(), include_audit=True)
    for purpose in (PURPOSE_MATRIX, PURPOSE_GT, PURPOSE_AUDIT):
        assert by[purpose] == c["executed"][purpose], purpose
    est = count_requests(p.cfg, p.plan, inc_slots={s: t.inc_slot for s, t in trajs.items()})
    distinct = (
        len({t.prompts[k] for t in trajs.values() for k in range(1, t.R + 1)})
        == len(trajs) * p.cfg.run.rounds
    )
    for purpose in (PURPOSE_MATRIX, PURPOSE_GT, PURPOSE_AUDIT):
        if distinct:  # the estimate assumes candidate prompts differ across seeds
            assert by[purpose] == est["by_purpose"][purpose]["executed"], purpose
        else:
            assert by[purpose] <= est["by_purpose"][purpose]["executed"], purpose
    assert by["trajectory_dev"] == len(p.dev_split())  # slot 0 executed once for all seeds


def test_stored_cells_physical_iff_distinct_generation(full_run) -> None:
    """In the STORED cells: a greedy rerun is a distinct generation from its creation cell iff it is flagged
    physical (nonce), every t02 round (full mode) and GT draw is its own generation, and audit cells never
    share a generation with any matrix cell (what the analyses rely on to tell measured from by-construction)."""
    p, _, _ = full_run
    cells = p.store.get_cells(p.run_id, split="test")
    by = {
        (c["seed"], c["decoding_id"], c["slot"], c["draw_kind"], c["draw"], c["item_idx"]): c for c in cells
    }
    n_phys = n_shared = 0
    for (s, d, k, kind, r, n), c in by.items():
        if kind != "round":
            continue
        creation = by[(s, d, k, "round", k, n)]["gen_key"]
        if d == "greedy" and r > k:
            assert (c["gen_key"] != creation) == bool(c["physical"])
            n_phys += c["physical"]
            n_shared += not c["physical"]
        elif d == "t02":
            assert c["physical"] == 1 and (r == k or c["gen_key"] != creation)
    assert n_phys and n_shared  # both kinds occur in the smoke run (ages mode)
    t02 = [c["gen_key"] for c in cells if c["decoding_id"] == "t02" and c["draw_kind"] in ("round", "gt")]
    assert len(t02) == len(set(t02))
    audit = {c["gen_key"] for c in cells if c["draw_kind"] == "audit"}
    assert audit and not audit & {c["gen_key"] for c in cells if c["draw_kind"] != "audit"}


# --------------------------------------------------------------------------- guards


def test_config_change_on_reopen_raises(full_run) -> None:
    p, _, _ = full_run
    before = (p.run_dir / "config.yaml").read_text()
    changed = load_config(SMOKE, ["audit.repeats=2"])
    with pytest.raises(ConfigMismatch):
        Pipeline(changed, p.run_dir, log=_quiet).open()
    assert (p.run_dir / "config.yaml").read_text() == before  # nothing overwritten
    flip = load_config(SMOKE, ["backend.mock.greedy_flip_rate=0.1"])  # also changes the engine fingerprint
    with pytest.raises(EngineMismatch):
        Pipeline(flip, p.run_dir, log=_quiet, allow_config_change=True).open()
    assert (p.run_dir / "config.yaml").read_text() == before


def test_allow_config_change_records_it(tmp_path: Path) -> None:
    cfg = load_config(SMOKE)
    with Pipeline(cfg, tmp_path / "r", log=_quiet) as p:
        p.run(["data"])
    changed = load_config(SMOKE, ["audit.repeats=2"])
    with Pipeline(changed, tmp_path / "r", log=_quiet, allow_config_change=True) as p:
        assert p.store.get_run(p.run_id)["config_hash"] == changed.config_hash()
        assert p.store.meta_items("config_change")
    assert yaml.safe_load((tmp_path / "r" / "config.yaml").read_text())["audit"]["repeats"] == 2


def test_frozen_plan_is_kept_on_reopen(tmp_path: Path) -> None:
    """Editing the analysis plan after a run started never silently replaces the run's frozen plan.yaml."""
    plan_src = tmp_path / "plan.yaml"
    plan_src.write_bytes((REPO_ROOT / "analysis_plans" / "prereg_v1.yaml").read_bytes())
    cfg = load_config(SMOKE, [f"analysis_plan={plan_src}"])
    run_dir = tmp_path / "r"
    with Pipeline(cfg, run_dir, log=_quiet) as p:
        p.run(["data"])
        frozen_hash = p.plan.plan_hash()
    frozen = (run_dir / "plan.yaml").read_bytes()
    data = yaml.safe_load(plan_src.read_text())
    data["bootstrap"]["B"] = int(data["bootstrap"]["B"]) + 1  # a post-hoc plan edit
    plan_src.write_text(yaml.safe_dump(data, sort_keys=False))
    msgs: list[str] = []
    with Pipeline(cfg, run_dir, log=msgs.append) as p:
        assert (run_dir / "plan.yaml").read_bytes() == frozen
        assert p.plan.plan_hash() == frozen_hash == p.store.get_run(p.run_id)["plan_hash"]
        assert not p.store.meta_items("plan_change")
        assert any("WARNING" in m and "frozen" in m for m in msgs)
    with Pipeline(cfg, run_dir, log=_quiet, allow_config_change=True) as p:  # deliberate, recorded change
        assert (run_dir / "plan.yaml").read_bytes() == plan_src.read_bytes()
        assert p.store.get_run(p.run_id)["plan_hash"] == p.plan.plan_hash() != frozen_hash
        assert p.store.meta_items("plan_change")


def test_status_is_read_only(tmp_path: Path) -> None:
    """status() on a closed pipeline opens nothing for writing: no run dir for a fresh path, no backend, no
    shard replay / quarantine (a concurrent writer's in-flight shard survives), no rewritten run-dir files."""
    nowhere = tmp_path / "nowhere"
    st = Pipeline(load_config(SMOKE), nowhere, log=_quiet).status()
    assert not nowhere.exists() and not st["complete"] and st["stages"]["data"]["done"] == 0
    run_dir = tmp_path / "r"
    with Pipeline(load_config(SMOKE), run_dir, log=_quiet) as p:
        p.run(["data", "trajectory"])
        last = int(p.store.get_meta("last_shard_seq"))
        before = p.status()
    in_flight = (
        run_dir / "shards" / f"{last + 1:06d}_matrix_0123abcd.jsonl.gz"
    )  # published, not yet committed
    in_flight.write_bytes(b"not committed yet")
    stamps = {f: (run_dir / f).stat().st_mtime_ns for f in ("config.yaml", "plan.yaml", "provenance.json")}
    q = Pipeline(load_config(SMOKE), run_dir, log=_quiet)
    st = q.status()
    assert q.store is None and q.backend is None and q.shard_writer is None
    assert in_flight.is_file() and not list((run_dir / "shards").glob("*.orphan"))
    assert {f: (run_dir / f).stat().st_mtime_ns for f in stamps} == stamps
    assert st["stages"] == before["stages"] and st["synthetic"]
    with Store(run_dir / "store.sqlite", read_only=True) as s:
        assert int(s.get_meta("last_shard_seq")) == last


def test_audit_config_is_checked_before_any_work(tmp_path: Path) -> None:
    """A bad audit config fails at once, not after hours of trajectory / matrix work; bad stage names too."""
    run_dir = tmp_path / "r"
    with pytest.raises(ValueError, match="unknown stage"):
        Pipeline(load_config(SMOKE), run_dir, log=_quiet).run(["data", "bogus"])
    bad_seed = load_config(SMOKE, ["run.seeds=[1, 2]"])  # audit.seed stays 0
    with pytest.raises(ValueError, match="audit.seed 0 is not a run seed"):
        Pipeline(bad_seed, run_dir, log=_quiet).run()
    greedy_only = load_config(
        SMOKE, ["environments={E1: {decoding: greedy, extractor: v1}, E3: {decoding: greedy, extractor: v2}}"]
    )
    with pytest.raises(ValueError, match="audit.decodings"):
        Pipeline(greedy_only, run_dir, log=_quiet).run(UP_TO_AUDIT)
    assert not run_dir.exists()  # nothing was touched
    with Pipeline(bad_seed, run_dir, log=_quiet) as p:  # stages without the audit still run
        assert p.run(["data"])["status"] == "ok"
        assert "audit.seed" in p.status()["stages"]["audit"]["error"]


def test_audit_without_items_counts_as_complete(tmp_path: Path) -> None:
    cfg = load_config(SMOKE, ["audit.n_items=0", "run.rounds=1"])
    with Pipeline(cfg, tmp_path / "r", log=_quiet) as p:
        res = p.run(UP_TO_AUDIT)
        assert res["status"] == "ok" and res["complete"]
        assert res["progress"]["stages"]["audit"] == {
            "enabled": True,
            "planned": 0,
            "done": 0,
            "complete": True,
        }
        assert not p.store.get_audit_results(p.run_id)
        assert not [r for r in p.store.ledger_rows(p.run_id) if r["purpose"] == "audit"]


def test_stage_validation_and_order(tmp_path: Path) -> None:
    p = Pipeline(load_config(SMOKE), tmp_path / "r", log=_quiet)
    with pytest.raises(ValueError, match="unknown stage"):
        p.run(["matrix", "bogus"])
    fresh = p.status()
    assert not fresh["complete"] and fresh["stages"]["data"]["done"] == 0 and fresh["executed"] == 0
    assert fresh["stages"]["audit"]["planned"] == 4 and not fresh["stages"]["audit"]["complete"]
    with pytest.raises(RuntimeError, match="trajectory"):
        p.run(["matrix"])
    res = p.run(["score", "data"])  # canonical order: data before score
    assert list(res["stages"]) == ["data", "score"] and res["stages"]["data"]["written"] == 48
    p.close()


def test_synthetic_flag_and_named_run(tmp_path: Path) -> None:
    with Pipeline(load_config(SMOKE), tmp_path / "r", run_id="custom", log=_quiet) as p:
        p.run(["data", "trajectory"])
        assert p.store.get_run("custom")["synthetic"] == 1
        assert {r["run_id"] for r in p.store.query("SELECT DISTINCT run_id FROM cells")} == {"custom"}
        assert p.status()["synthetic"]


# --------------------------------------------------------------------------- analyze hand-off


def test_analyze_skipped_when_modules_missing(full_run, monkeypatch: pytest.MonkeyPatch) -> None:
    p, _, _ = full_run
    msgs: list[str] = []
    monkeypatch.setitem(sys.modules, "driftlab.analysis.bundle", None)  # import -> ImportError
    with Pipeline(p.cfg, p.run_dir, log=msgs.append) as q:
        res = q.run(["analyze"])
        assert res["status"] == "ok" and "skipped" in res["stages"]["analyze"]
        assert any("WARNING" in m and "analyze" in m for m in msgs)
        assert q.status()["complete"] and not q.status()["stages"]["analyze"]["complete"]
        assert {r["stage"]: r["status"] for r in q.store.get_stage_status(q.run_id)}["analyze"] == "skipped"


def _stub_analysis(monkeypatch: pytest.MonkeyPatch, calls: list[tuple], with_tables: bool) -> None:
    bundle_mod = types.ModuleType("driftlab.analysis.bundle")

    def analyze(run_dir, **kw):
        calls.append(("analyze", Path(run_dir)))
        kw.get("log", print)("[analyze] stub")  # the pipeline must hand over its own log
        out = Path(run_dir) / "exports" / "analysis"
        out.mkdir(parents=True, exist_ok=True)
        (out / "bundle.json").write_text("{}")
        return "BUNDLE"

    bundle_mod.analyze = analyze  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "driftlab.analysis.bundle", bundle_mod)
    if not with_tables:
        monkeypatch.setitem(sys.modules, "driftlab.reporting.tables", None)  # import -> ImportError
        return
    tables_mod = types.ModuleType("driftlab.reporting.tables")

    def write_tables(bundle, out_dir):
        calls.append(("tables", bundle, Path(out_dir)))

    tables_mod.write_tables = write_tables  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "driftlab.reporting.tables", tables_mod)


def test_analyze_calls_bundle_and_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple] = []
    msgs: list[str] = []
    _stub_analysis(monkeypatch, calls, with_tables=True)
    with Pipeline(load_config(SMOKE), tmp_path / "r", log=msgs.append) as q:
        assert not q.status()["stages"]["analyze"]["complete"]
        res = q.run(["analyze"])
        assert "[analyze] stub" in msgs  # analyze() logs through the pipeline's log (quiet stays quiet)
        assert calls == [("analyze", q.run_dir), ("tables", "BUNDLE", q.run_dir / "exports" / "tables")]
        assert res["stages"]["analyze"]["tables_dir"].endswith("tables")
        assert q.status()["stages"]["analyze"]["complete"]
        assert {r["stage"]: r["status"] for r in q.store.get_stage_status(q.run_id)}["analyze"] == "complete"


def test_analyze_without_reporting_still_builds_the_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple] = []
    msgs: list[str] = []
    _stub_analysis(monkeypatch, calls, with_tables=False)
    with Pipeline(load_config(SMOKE), tmp_path / "r", log=msgs.append) as q:
        res = q.run(["analyze"])
        assert calls == [("analyze", q.run_dir)] and "tables_skipped" in res["stages"]["analyze"]
        assert any("WARNING" in m and "tables" in m for m in msgs)
        assert q.status()["stages"]["analyze"]["complete"]  # the bundle exists
        assert {r["stage"]: r["status"] for r in q.store.get_stage_status(q.run_id)}["analyze"] == "partial"


# --------------------------------------------------------------------------- wrapper


def test_run_experiment_wrapper(tmp_path: Path, full_run) -> None:
    p, _, _ = full_run
    res = run_experiment(SMOKE, tmp_path / "wrapped", stages=UP_TO_AUDIT)
    assert res["status"] == "ok" and res["complete"] and res["run_dir"] == str(tmp_path / "wrapped")
    with Store(tmp_path / "wrapped" / "store.sqlite") as s:
        assert s.content_digest("smoke_mock") == p.store.content_digest(p.run_id)
    res2 = run_experiment(SMOKE, tmp_path / "wrapped", stages=UP_TO_AUDIT)
    assert res2["executed"] == 0 and res2["complete"]
    other = run_experiment(
        SMOKE, tmp_path / "other", overrides=["run.rounds=2"], stages=["data", "trajectory"]
    )
    assert other["progress"]["stages"]["trajectory"] == {
        "planned": 4,
        "done": 4,
        "complete": True,
        "per_seed": {0: 2, 1: 2},
    }


def test_bad_audit_seed_is_reported(tmp_path: Path) -> None:
    p = Pipeline(load_config(SMOKE, ["audit.seed=7"]), tmp_path / "r", log=_quiet)
    p.run(["data", "trajectory"])
    st = p.status()["stages"]["audit"]
    assert st["complete"] is False and "audit.seed" in st["error"]
    with pytest.raises(ValueError, match="audit.seed"):
        p.run(["audit"])
    p.close()


@pytest.mark.slow
def test_full_size_demo_executes_the_estimated_counts(tmp_path: Path) -> None:
    """demo_mock (3 seeds x 11 rounds x 200 items, ~1 min): ledger == planner == estimate for these trajectories."""
    cfg = load_config(REPO_ROOT / "configs" / "demo_mock.yaml")
    with Pipeline(cfg, tmp_path / "demo", log=_quiet) as p:
        res = p.run(UP_TO_AUDIT)
        assert res["status"] == "ok" and res["complete"] and res["synthetic"]
        trajs = p.trajectories()
        by: dict[str, int] = {}
        for r in p.store.ledger_rows(p.run_id):
            by[r["purpose"]] = by.get(r["purpose"], 0) + r["n_executed"]
        c = count_plan(cfg, p.plan, trajs, p.eval_split(), include_audit=True)
        est = count_requests(cfg, p.plan, inc_slots={s: t.inc_slot for s, t in trajs.items()})
        for purpose in (PURPOSE_MATRIX, PURPOSE_GT):
            assert by[purpose] == c["executed"][purpose] <= est["by_purpose"][purpose]["executed"]
        assert by["candidate_dev"] <= est["by_purpose"]["candidate_dev"]["executed"]
        assert by["trajectory_dev"] == est["by_purpose"]["trajectory_dev"]["executed"] == 200
        assert by[PURPOSE_AUDIT] == c["executed"][PURPOSE_AUDIT]
        greedy = [r for r in p.store.get_audit_results(p.run_id) if r["decoding_id"] == "greedy"]
        assert greedy and all(90.0 <= r["pct_text_identical"] <= 100.0 for r in greedy)  # flip rate 0.004
