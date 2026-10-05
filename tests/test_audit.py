"""Determinism audit: nonce-forced regenerations vs creation cells, shuffled/rechunked, v1 + v2 flip rates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from driftlab.audit import AUDIT_COLUMNS, audit_chunk_size, run_audit
from driftlab.backends.mock import MockBackend
from driftlab.config import REPO_ROOT, load_config
from driftlab.engine import GenerationEngine
from driftlab.extraction import extract, is_correct
from driftlab.pipeline import Pipeline
from driftlab.planning import audit_slots, audit_tasks
from driftlab.store import Store, replay_shards

SMOKE = REPO_ROOT / "configs" / "smoke_mock.yaml"


def _pipeline(
    tmp: Path, overrides: list[str] | None = None, stages=("data", "trajectory", "matrix")
) -> Pipeline:
    p = Pipeline(load_config(SMOKE, overrides), tmp / "run", log=lambda _m: None)
    p.run(list(stages))
    return p


def _engine(p: Pipeline, chunk_size: int = 5000) -> GenerationEngine:
    return GenerationEngine(p.backend, p.store, p.run_id, chunk_size=chunk_size, shard_writer=p.shard_writer)


def _audit(p: Pipeline, engine: GenerationEngine | None = None) -> list[dict]:
    return run_audit(
        p.cfg, engine or _engine(p), p.store, p.run_id, p.trajectories(), p.eval_split(), log=lambda _m: None
    )


def test_flip_rate_zero_gives_identical_text(tmp_path: Path) -> None:
    p = _pipeline(tmp_path, ["audit.repeats=2"])
    eng = _engine(p)
    rows = _audit(p, eng)
    traj = p.trajectories()[0]
    slots = audit_slots(p.cfg, traj)
    assert len(slots) == 2 and slots[0] == 0 and slots[1] == traj.incumbent_after(traj.R)
    assert len(rows) == len(slots) * 2 * 2
    assert {(r["decoding_id"], r["slot"], r["repeat"]) for r in rows} == {
        (d, k, a) for d in ("greedy", "t02") for k in slots for a in (0, 1)
    }
    for r in rows:
        assert r["n"] == 24 and r["seed"] == 0
        assert r["pct_text_identical"] == 100.0
        assert r["pct_correct_flip"] == r["pct_correct_flip_v1"] == r["pct_correct_flip_v2"] == 0.0
    stored = p.store.get_audit_results(p.run_id)
    assert sorted(stored, key=lambda x: (x["decoding_id"], x["slot"], x["repeat"])) == sorted(
        ({c: r[c] for c in AUDIT_COLUMNS} for r in rows),
        key=lambda x: (x["decoding_id"], x["slot"], x["repeat"]),
    )
    # every audit cell is a fresh physical generation (nonce) and t02 reuses the creation cell's seed
    n_tasks = len(rows) * 24
    assert eng.n_executed == n_tasks
    audit = [x for x in p.store.ledger_rows(p.run_id) if x["purpose"] == "audit"]
    assert sum(x["n_executed"] for x in audit) == n_tasks and {x["stage"] for x in audit} == {"audit"}
    cells = p.store.query(
        "SELECT a.decoding_id, a.draw, a.physical, ga.nonce, ga.seed AS aseed, gc.seed AS cseed, a.gen_key AS ak, "
        "c.gen_key AS ck FROM cells a JOIN cells c ON c.run_id = a.run_id AND c.seed = a.seed AND c.split = a.split "
        "AND c.decoding_id = a.decoding_id AND c.slot = a.slot AND c.draw_kind = 'round' AND c.draw = a.slot "
        "AND c.item_idx = a.item_idx JOIN generations ga ON ga.gen_key = a.gen_key "
        "JOIN generations gc ON gc.gen_key = c.gen_key WHERE a.draw_kind = 'audit'"
    )
    assert len(cells) == n_tasks
    for c in cells:
        assert c["physical"] == 1 and c["nonce"] == f"audit:s0:{c['draw']}" and c["ak"] != c["ck"]
        if c["decoding_id"] == "t02":
            assert c["aseed"] == c["cseed"] is not None
        else:
            assert c["aseed"] is None
    # re-running is all cache hits and converges to the same rows
    eng2 = _engine(p)
    assert _audit(p, eng2) == rows and eng2.n_executed == 0


def test_flip_rate_positive_shows_greedy_differences(tmp_path: Path) -> None:
    p = _pipeline(tmp_path, ["backend.mock.greedy_flip_rate=0.5"])
    rows = _audit(p)
    greedy = [r for r in rows if r["decoding_id"] == "greedy"]
    t02 = [r for r in rows if r["decoding_id"] == "t02"]
    assert greedy and t02
    assert any(r["pct_text_identical"] < 100.0 for r in greedy)
    assert any(r["pct_correct_flip"] > 0.0 for r in greedy)
    assert all(r["pct_correct_flip_v2"] <= 100.0 for r in greedy)
    assert all(
        r["pct_text_identical"] == 100.0 for r in t02
    )  # same seed: the mock's sampling is reproducible


def test_counts_match_an_independent_recomputation(tmp_path: Path) -> None:
    """Identical-text and v1/v2 flip counts re-derived from the stored audit and creation cells."""
    p = _pipeline(tmp_path, ["backend.mock.greedy_flip_rate=0.3", "audit.repeats=2"])
    rows = _audit(p)
    gold = {it.idx: it.gold for it in p.eval_split()}
    seen_flip = False
    for r in rows:
        audit = p.store.get_cell_texts(p.run_id, 0, "test", r["decoding_id"], r["slot"], "audit", r["repeat"])
        base = p.store.get_cell_texts(p.run_id, 0, "test", r["decoding_id"], r["slot"], "round", r["slot"])
        assert [x["item_idx"] for x in audit] == [x["item_idx"] for x in base] == list(range(24))
        pairs = [
            (a["response"], b["response"], gold[a["item_idx"]]) for a, b in zip(audit, base, strict=True)
        ]

        def ok(x: str, t: str, g: str) -> bool:
            return is_correct(extract(x, t), g)

        ident = sum(a == b for a, b, _ in pairs)
        f1 = sum(ok("v1", a, g) != ok("v1", b, g) for a, b, g in pairs)
        f2 = sum(ok("v2", a, g) != ok("v2", b, g) for a, b, g in pairs)
        assert (r["n_text_identical"], r["n_flip_v1"], r["n_flip_v2"], r["n"]) == (ident, f1, f2, 24)
        assert r["pct_text_identical"] == pytest.approx(100 * ident / 24)
        assert r["pct_correct_flip"] == r["pct_correct_flip_v1"] == pytest.approx(100 * f1 / 24)
        assert r["pct_correct_flip_v2"] == pytest.approx(100 * f2 / 24)
        seen_flip |= f1 > 0
    assert seen_flip  # the recomputation is not vacuous


def test_audit_results_reach_the_shard_log(tmp_path: Path) -> None:
    p = _pipeline(tmp_path)
    _audit(p)
    fresh = Store(tmp_path / "fresh.sqlite")
    replay_shards(fresh, p.shard_dir)
    assert fresh.get_audit_results(p.run_id) == p.store.get_audit_results(p.run_id) != []
    fresh.close()


def test_audit_decoding_outside_the_matrix_is_refused(tmp_path: Path) -> None:
    greedy_only = (
        "environments={E1: {decoding: greedy, extractor: v1}, E3: {decoding: greedy, extractor: v2}}"
    )
    p = _pipeline(tmp_path, [greedy_only])
    with pytest.raises(ValueError, match="audit.decodings"):
        _audit(p)
    # no t02 creation cells were injected into the cube
    assert {c["decoding_id"] for c in p.store.get_cells(p.run_id, split="test")} == {"greedy"}
    ok = _pipeline(tmp_path / "ok", [greedy_only, "audit.decodings=[greedy]"])
    rows = _audit(ok)
    assert {r["decoding_id"] for r in rows} == {"greedy"}
    assert {c["decoding_id"] for c in ok.store.get_cells(ok.run_id, split="test")} == {"greedy"}


def test_requests_are_shuffled_and_rechunked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = _pipeline(tmp_path)
    seen: list[list[tuple]] = []
    original = MockBackend.generate

    def recording(self, reqs):
        seen.append([(r.system, r.user, r.decoding.id, r.nonce) for r in reqs])
        return original(self, reqs)

    monkeypatch.setattr(MockBackend, "generate", recording)
    eng = _engine(p, chunk_size=30)
    _audit(p, eng)
    assert eng.chunk_size == 30  # restored
    canonical = [
        (t.request.system, t.request.user, t.request.decoding.id, t.request.nonce)
        for t in audit_tasks(p.cfg, p.trajectories(), p.eval_split(), run_id=p.run_id)
    ]
    flat = [x for chunk in seen for x in chunk]
    assert sorted(flat) == sorted(canonical) and flat != canonical
    assert [len(c) for c in seen] == [audit_chunk_size(30)] * (len(canonical) // 20) + (
        [len(canonical) % 20] if len(canonical) % 20 else []
    )
    assert audit_chunk_size(30) == 20 and audit_chunk_size(1) == 1 and audit_chunk_size(2) == 1


def test_static_trajectory_audits_one_slot(tmp_path: Path) -> None:
    p = _pipeline(tmp_path, ["trajectory.mode=static"])
    rows = _audit(p)
    assert {r["slot"] for r in rows} == {0} and len(rows) == 2


def test_missing_creation_cells_are_generated(tmp_path: Path) -> None:
    p = _pipeline(tmp_path, stages=("data", "trajectory"))
    rows = _audit(p)
    assert all(r["pct_text_identical"] == 100.0 for r in rows)
    ledger = p.store.ledger_rows(p.run_id)
    made = [x for x in ledger if x["purpose"] == "eval_matrix"]
    assert made and {x["stage"] for x in made} == {"matrix"}
    # creation cells are generated BEFORE the audit regenerations (the audit draws are always later draws)
    assert max(x["id"] for x in made) < min(x["id"] for x in ledger if x["purpose"] == "audit")
    res = p.run(["matrix", "score"])  # the matrix later re-uses those creation cells without conflicts
    assert res["status"] == "ok" and res["progress"]["stages"]["matrix"]["complete"]


def test_eval_split_type_guard(tmp_path: Path) -> None:
    p = _pipeline(tmp_path, stages=("data", "trajectory"))
    with pytest.raises(TypeError, match="EVAL"):
        run_audit(p.cfg, _engine(p), p.store, p.run_id, p.trajectories(), p.dev_split())  # type: ignore[arg-type]


def test_pipeline_audit_stage_scores_and_skips_when_complete(tmp_path: Path) -> None:
    p = _pipeline(tmp_path, stages=("data", "trajectory", "matrix", "score", "audit"))
    st = p.status()
    assert st["stages"]["audit"] == {"enabled": True, "planned": 4, "done": 4, "complete": True}
    assert st["stages"]["score"]["unscored"] == 0  # audit generations scored by the audit stage
    res = p.run(["audit"])
    assert res["stages"]["audit"]["skipped"] == "complete" and res["executed"] == 0
    off = Pipeline(load_config(SMOKE, ["audit.enabled=false"]), tmp_path / "off", log=lambda _m: None)
    res = off.run(["data", "trajectory", "matrix", "score", "audit"])
    assert res["stages"]["audit"] == {"skipped": "audit disabled"} and res["complete"]
    assert not off.store.get_audit_results(off.run_id)
    assert json.loads(json.dumps(res["progress"]))["stages"]["audit"]["enabled"] is False
