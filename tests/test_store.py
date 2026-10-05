"""Experiment store: idempotence, integrity guards, snapshots, shard replay and scoring joins."""

from __future__ import annotations

import gzip
import json
import sqlite3
import threading
from pathlib import Path

import numpy as np
import pytest

from driftlab.keys import sha256_text
from driftlab.store import (
    ConfigMismatch,
    EngineMismatch,
    ShardCorrupt,
    ShardWriter,
    SlotConflict,
    Store,
    StoreError,
    list_shards,
    read_shard,
    replay_shards,
)

RUN = "run_test"


# ---------------------------------------------------------------- helpers (fixtures stay in this file)
@pytest.fixture
def store(tmp_path: Path):
    s = Store(tmp_path / "db" / "driftlab.sqlite")
    yield s
    s.close()


def _gen(key: str, response: str | None = None, **over: object) -> dict:
    row = {
        "gen_key": key,
        "engine_fp": "eng",
        "model_id": "fake/model",
        "model_revision": "r1",
        "rendered_sha": sha256_text("rendered:" + key),
        "system_hash": sha256_text("sys"),
        "user_hash": sha256_text("user:" + key),
        "decoding_json": json.dumps({"id": "greedy", "temperature": 0.0}),
        "seed": None,
        "nonce": None,
        "response": response if response is not None else f"The answer is \\boxed{{{len(key)}}}",
        "finish_reason": "stop",
        "n_prompt_tokens": 10,
        "n_completion_tokens": 5,
        "latency_ms": 1.5,
        "batch_id": "test:0",
        "created_at": "2026-01-01T00:00:00.000Z",
    }
    row.update(over)
    return row


def _cell(key: str, item: int, *, seed: int = 0, slot: int = 0, draw: int = 0, split: str = "test", **over):
    row = {
        "run_id": RUN,
        "seed": seed,
        "split": split,
        "decoding_id": "greedy",
        "slot": slot,
        "draw_kind": "round",
        "draw": draw,
        "item_idx": item,
        "gen_key": key,
        "physical": 1,
    }
    row.update(over)
    return row


def _ledger(n_req: int, n_exec: int, **over) -> dict:
    row = {
        "run_id": RUN,
        "seed": 0,
        "stage": "matrix",
        "purpose": "eval_matrix",
        "round": 0,
        "n_requested": n_req,
        "n_executed": n_exec,
        "n_cache_hits": n_req - n_exec,
        "prompt_tokens": 10 * n_exec,
        "completion_tokens": 5 * n_exec,
        "wall_s": 0.01,
    }
    row.update(over)
    return row


def _chunk(prefix: str, n: int, *, draw: int = 0) -> tuple[list[dict], list[dict], dict]:
    gens = [_gen(f"{prefix}{i:03d}") for i in range(n)]
    cells = [_cell(g["gen_key"], i, draw=draw) for i, g in enumerate(gens)]
    return gens, cells, _ledger(n, n, round=draw)


def _ensure(store: Store, **over) -> dict:
    kw: dict = {
        "config_json": {"run": {"name": "t"}},
        "config_hash": "cfg1",
        "plan_hash": "plan1",
        "provenance_json": {"python": "3.11"},
        "engine_fp": "eng1",
        "synthetic": False,
    }
    kw.update(over)
    return store.ensure_run(RUN, **kw)


def _items(n: int = 5) -> list[dict]:
    return [
        {"idx": i, "question": f"Q{i}?", "answer_text": f"... #### {i * 10}", "gold": str(i * 10)}
        for i in range(n)
    ]


# ---------------------------------------------------------------- basics
def test_open_sets_wal_schema_and_version(store: Store, tmp_path: Path):
    assert store.path == tmp_path / "db" / "driftlab.sqlite"
    assert store.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert store.get_meta("schema_version") == "1"
    assert store.count_rows("generations") == 0
    assert store.quick_check()
    # reopening is idempotent
    store.close()
    again = Store(tmp_path / "db" / "driftlab.sqlite")
    assert again.get_meta("schema_version") == "1"
    again.close()


def test_schema_version_mismatch_is_refused(tmp_path: Path):
    p = tmp_path / "x.sqlite"
    s = Store(p)
    s.set_meta("schema_version", "99")
    s.close()
    with pytest.raises(StoreError):
        Store(p)


def test_meta_roundtrip_and_transaction_rollback(store: Store):
    assert store.get_meta("nope") is None
    store.set_meta("a", 1)
    assert store.get_meta("a") == "1"
    with pytest.raises(RuntimeError), store.transaction():
        store.set_meta("a", 2)
        store.set_meta("b", 3)
        raise RuntimeError("boom")
    assert store.get_meta("a") == "1" and store.get_meta("b") is None
    with store.transaction():
        with store.transaction():  # nested joins the outer transaction
            store.set_meta("c", "x")
        assert store.in_transaction
    assert store.get_meta("c") == "x" and not store.in_transaction


def test_read_only_store_rejects_writes(store: Store, tmp_path: Path):
    store.put_items("test", _items(2))
    ro = Store(store.path, read_only=True)
    assert len(ro.get_items("test")) == 2
    with pytest.raises(sqlite3.OperationalError):
        ro.set_meta("k", "v")
    ro.close()
    with pytest.raises(FileNotFoundError):
        Store(tmp_path / "missing.sqlite", read_only=True)


def test_unknown_table_and_column_rejected(store: Store):
    with pytest.raises(ValueError):
        store.count_rows("nope; DROP TABLE runs")
    with pytest.raises(ValueError):
        store.write_chunk([{**_gen("k1"), "bogus": 1}], [], None)


def test_constraint_violations_raise_instead_of_being_ignored(store: Store):
    with pytest.raises(sqlite3.IntegrityError):  # draw_kind CHECK
        store.write_chunk([_gen("k1")], [_cell("k1", 0, draw_kind="bogus")], None)
    with pytest.raises(sqlite3.IntegrityError):  # NOT NULL response
        store.write_chunk([_gen("k2", response=None) | {"response": None}], [], None)
    assert store.count_rows("generations") == 0  # whole chunk rolled back


# ---------------------------------------------------------------- idempotence
def test_idempotent_inserts(store: Store):
    store.put_items("test", _items())
    store.put_items("test", _items())
    assert [r["idx"] for r in store.get_items("test")] == [0, 1, 2, 3, 4]
    assert store.get_items("test")[2] == {
        "split": "test",
        "idx": 2,
        "question": "Q2?",
        "answer_text": "... #### 20",
        "gold": "20",
    }
    h1 = store.put_prompt("Solve it.")
    h2 = store.put_prompt("Solve it.")
    assert h1 == h2 == sha256_text("Solve it.")
    assert store.get_prompt(h1) == "Solve it."
    assert store.get_prompt("0" * 64) is None
    assert store.get_prompts([h1, "0" * 64]) == {h1: "Solve it."}

    gens, cells, ledger = _chunk("g", 4)
    store.write_chunk(gens, cells, ledger)
    store.write_chunk(gens, cells, None)
    # a cache-served re-write with a different response does not overwrite the stored generation
    store.write_chunk([_gen("g000", response="different")], [], None)
    assert store.count_rows("generations") == 4
    assert store.count_rows("cells") == 4
    assert store.count_rows("ledger") == 1
    assert store.lookup_generations(["g000"])["g000"]["response"] == gens[0]["response"]


def test_proposals_and_rounds_replace(store: Store):
    row = {
        "run_id": RUN,
        "seed": 0,
        "round": 1,
        "attempt": 0,
        "meta_prompt_hash": "m",
        "error_item_idxs": [3, 1, 4],
        "gen_key": "g",
        "parsed_prompt_hash": None,
        "valid": False,
        "violations": ["too_short"],
    }
    store.put_proposal(row)
    store.put_proposal({**row, "valid": True, "violations": []})
    (p,) = store.get_proposals(RUN, seed=0, round=1)
    assert (
        p["valid"] == 1
        and json.loads(p["violations"]) == []
        and json.loads(p["error_item_idxs"]) == [3, 1, 4]
    )
    assert store.get_proposals(RUN, seed=1) == []

    tr = {
        "run_id": RUN,
        "seed": 0,
        "round": 1,
        "incumbent_slot": 0,
        "candidate_slot": 1,
        "inc_dev_acc": 0.5,
        "cand_dev_acc": np.float64(0.55),
        "advanced": np.bool_(True),
        "n_attempts": np.int64(1),
        "is_fallback": 0,
    }
    store.put_trajectory_round(tr)
    store.put_trajectory_round({**tr, "cand_dev_acc": 0.6})
    (r,) = store.get_trajectory_rounds(RUN)
    assert r["cand_dev_acc"] == 0.6 and r["advanced"] == 1 and r["completed_at"]


def test_slot_determinism_guard(store: Store):
    h = store.put_slot(RUN, 0, 0, "Initial prompt.", 0, None, "initial")
    assert store.put_slot(RUN, 0, 0, "Initial prompt.", 0, None, "initial") == h
    with pytest.raises(SlotConflict):
        store.put_slot(RUN, 0, 0, "A different prompt.", 0, None, "initial")
    # the failed write rolled back completely (prompt row too)
    assert store.get_prompt(sha256_text("A different prompt.")) is None
    store.put_slot(RUN, 0, 1, "Candidate one.", 1, 0, "proposer")
    store.put_slot(RUN, 1, 0, "Initial prompt.", 0, None, "initial")
    slots = store.get_slots(RUN, seed=0)
    assert [(s["slot"], s["prompt_text"], s["parent_slot"]) for s in slots] == [
        (0, "Initial prompt.", None),
        (1, "Candidate one.", 0),
    ]
    assert len(store.get_slots(RUN)) == 3
    with pytest.raises(sqlite3.IntegrityError):  # origin CHECK constraint
        store.put_slot(RUN, 0, 2, "x", 2, 1, "bogus")


# ---------------------------------------------------------------- runs
def test_ensure_run_create_and_get(store: Store):
    assert store.get_run() is None
    row = _ensure(store)
    assert row["run_id"] == RUN and row["config_hash"] == "cfg1" and row["synthetic"] == 0
    assert json.loads(row["config_json"]) == {"run": {"name": "t"}}
    assert store.get_run() == row == store.get_run(RUN)
    assert store.get_run("other") is None
    store.ensure_run("other", config_json="{}", config_hash="c", synthetic=False)
    assert len(store.list_runs()) == 2
    with pytest.raises(LookupError):
        store.get_run()


def test_synthetic_is_sticky(store: Store):
    _ensure(store, synthetic=True)
    assert _ensure(store, synthetic=False)["synthetic"] == 1
    assert store.get_run(RUN)["synthetic"] == 1
    store.ensure_run("r2", config_json="{}", config_hash="c", synthetic=False)
    assert store.ensure_run("r2", config_json="{}", config_hash="c", synthetic=True)["synthetic"] == 1
    assert store.ensure_run("r2", config_json="{}", config_hash="c", synthetic=False)["synthetic"] == 1


def test_config_mismatch(store: Store):
    _ensure(store)
    with pytest.raises(ConfigMismatch):
        _ensure(store, config_hash="cfg2")
    assert store.get_run(RUN)["config_hash"] == "cfg1"
    row = _ensure(store, config_hash="cfg2", config_json={"run": {"name": "t2"}}, allow_config_change=True)
    assert row["config_hash"] == "cfg2" and json.loads(row["config_json"])["run"]["name"] == "t2"
    assert json.loads(store.get_meta("config_change:0")) == {"run_id": RUN, "old": "cfg1", "new": "cfg2"}


def test_engine_mismatch(store: Store):
    _ensure(store, engine_fp=None)
    assert _ensure(store, engine_fp="eng1")["engine_fp"] == "eng1"  # first fingerprint is adopted
    assert _ensure(store, engine_fp=None)["engine_fp"] == "eng1"
    with pytest.raises(EngineMismatch):
        _ensure(store, engine_fp="eng2")
    assert _ensure(store, engine_fp="eng2", allow_engine_change=True)["engine_fp"] == "eng2"
    assert _ensure(store, engine_fp="eng3", allow_engine_change=True)["engine_fp"] == "eng3"
    events = store.meta_items("engine_change:")
    assert [json.loads(v)["new"] for v in events.values()] == ["eng2", "eng3"]
    assert list(events) == ["engine_change:0", "engine_change:1"]


def test_plan_hash_follows_latest_and_provenance_is_kept(store: Store):
    _ensure(store)
    row = _ensure(store, plan_hash="plan2", provenance_json={"python": "3.12"})
    assert row["plan_hash"] == "plan2"
    assert json.loads(row["provenance_json"]) == {"python": "3.11"}
    assert json.loads(store.get_meta("plan_change:0"))["old"] == "plan1"


# ---------------------------------------------------------------- generations, cells, ledger
def test_lookup_generations_batches_beyond_sql_limit(store: Store):
    gens = [_gen(f"k{i:05d}") for i in range(2100)]
    store.write_chunk(gens, [], None)
    found = store.lookup_generations([g["gen_key"] for g in gens] + ["missing"])
    assert len(found) == 2100 and "missing" not in found
    assert found["k01234"]["response"] == gens[1234]["response"]
    assert store.lookup_generations([]) == {}


def test_cells_queries(store: Store):
    gens, cells, ledger = _chunk("a", 3)
    cells.append(_cell("a000", 0, seed=1, split="train", slot=2, draw=2))
    store.write_chunk(gens, cells, ledger)
    keys = store.existing_cell_keys(RUN)
    assert (0, "test", "greedy", 0, "round", 0, 2) in keys and len(keys) == 4
    assert store.existing_cell_keys(RUN, split="train") == {(1, "train", "greedy", 2, "round", 2, 0)}
    assert len(store.existing_cell_keys(RUN, seed=0)) == 3
    assert store.existing_cell_keys("other") == set()
    rows = store.get_cells(RUN, split="test")
    assert [r["item_idx"] for r in rows] == [0, 1, 2]
    assert set(rows[0]) == {
        "run_id", "seed", "split", "decoding_id", "slot", "draw_kind", "draw", "item_idx", "gen_key", "physical",
    }  # fmt: skip
    texts = store.get_cell_texts(RUN, 0, "test", "greedy", 0, "round", 0)
    assert [t["item_idx"] for t in texts] == [0, 1, 2]
    assert texts[1] == {
        "item_idx": 1,
        "gen_key": "a001",
        "response": gens[1]["response"],
        "finish_reason": "stop",
    }
    (led,) = store.ledger_rows(RUN)
    assert led["n_requested"] == 3 and led["n_executed"] == 3 and led["created_at"]


def test_stage_status_audit_analysis_interactive(store: Store):
    store.set_stage_status(RUN, "matrix", None, 100, 10, "running")
    store.set_stage_status(RUN, "matrix", None, 100, 100, "done")
    store.set_stage_status(RUN, "trajectory", 0, 11, 11, "done")
    st = store.get_stage_status(RUN)
    assert [(r["stage"], r["seed"], r["n_done"], r["status"]) for r in st] == [
        ("matrix", -1, 100, "done"),
        ("trajectory", 0, 11, "done"),
    ]
    row = {"run_id": RUN, "decoding_id": "greedy", "slot": 0, "repeat": 0, "pct_text_identical": 1.0}
    store.put_audit_results([row, {**row, "repeat": 1, "pct_text_identical": 0.99, "n": 200}])
    store.put_audit_results([{**row, "pct_text_identical": 0.5}])
    audit = store.get_audit_results(RUN)
    assert [(a["repeat"], a["pct_text_identical"]) for a in audit] == [(0, 0.5), (1, 0.99)]
    payload = {"rows": [{"a": np.int64(3), "b": np.float64(0.5), "c": float("nan")}], "arr": np.arange(3)}
    store.put_analysis_result("plan1", "allpairs", payload)
    got = store.get_analysis_result("plan1", "allpairs")
    assert got["rows"][0]["a"] == 3 and got["arr"] == [0, 1, 2] and got["rows"][0]["c"] != got["rows"][0]["c"]
    assert store.get_analysis_result("plan1", "nope") is None
    store.put_analysis_result("plan1", "allpairs", [1, 2])
    assert store.get_analysis_result("plan1", "allpairs") == [1, 2]
    eid = store.put_interactive_event(backend="mock", seed=0, slot=1, env_id="E2", item_idx=3, note="hi")
    eid2 = store.put_interactive_event(backend="mock")
    assert eid2 == eid + 1
    with pytest.raises(ValueError):
        store.put_interactive_event(note="no backend")


# ---------------------------------------------------------------- scores
def test_unscored_joins_gold_and_respects_ext_hash(store: Store):
    store.put_items("test", _items(3))  # gold "0", "10", "20"
    store.put_items("train", [{"idx": 0, "question": "T?", "answer_text": "#### 7", "gold": "7"}])
    gens = [_gen("ga"), _gen("gb"), _gen("gc"), _gen("proposer_only")]
    cells = [
        _cell("ga", 0),
        _cell("gb", 1),
        _cell("gb", 1, draw=1),  # same generation, second cell (cache hit) -> scored once
        _cell("gc", 2),
        _cell("gc", 0, split="train"),  # same generation shown on a different split/item -> second gold
    ]
    store.write_chunk(gens, cells, None)
    todo = store.unscored("v1", "h1")
    assert [(r["gen_key"], r["gold"]) for r in todo] == [("ga", "0"), ("gb", "10"), ("gc", "20"), ("gc", "7")]
    assert todo[0]["response"] == gens[0]["response"]
    assert len(store.unscored("v1", "h1", limit=2)) == 2

    def score(key: str, gold: str, ext_hash: str = "h1", correct: int = 1) -> dict:
        return {
            "gen_key": key,
            "extractor": "v1",
            "ext_hash": ext_hash,
            "extracted": gold,
            "method": "boxed",
            "span_start": 0,
            "span_end": 3,
            "gold": gold,
            "correct": correct,
        }

    store.put_scores([score("ga", "0"), score("gb", "10")])
    store.put_scores([score("ga", "0", correct=0)])  # same ext_hash: ignored (idempotent)
    assert [(r["gen_key"], r["gold"]) for r in store.unscored("v1", "h1")] == [("gc", "20"), ("gc", "7")]
    assert len(store.unscored("v2", "h9")) == 4  # other extractor: nothing scored yet
    assert store.get_scores(["ga"])[0]["correct"] == 1
    # a changed extractor hash re-opens the generation and the new score replaces the stale one
    assert {r["gen_key"] for r in store.unscored("v1", "h2")} == {"ga", "gb", "gc"}
    store.put_scores([score("ga", "0", ext_hash="h2", correct=0)])
    (s,) = store.get_scores(["ga"], extractor="v1")
    assert s["ext_hash"] == "h2" and s["correct"] == 0
    assert store.get_scores(["ga"], extractor="v2") == []


# ---------------------------------------------------------------- digests & snapshots
def test_content_digest_ignores_timestamps_and_order(tmp_path: Path):
    a, b = Store(tmp_path / "a.sqlite"), Store(tmp_path / "b.sqlite")
    gens, cells, ledger = _chunk("d", 5)
    a.write_chunk(gens, cells, ledger)
    b.write_chunk(
        [{**g, "created_at": "2030-01-01", "latency_ms": 99.0, "batch_id": "x:7"} for g in reversed(gens)],
        list(reversed(cells)),
        _ledger(5, 0, wall_s=123.0),
    )
    assert a.content_digest(RUN) == b.content_digest(RUN)
    b.write_chunk([], [_cell("d000", 0, draw=1)], None)
    assert a.content_digest(RUN) != b.content_digest(RUN)
    a.close()
    b.close()


def test_backup_to_is_readable_and_equal(store: Store, tmp_path: Path):
    _ensure(store)
    store.put_items("test", _items())
    gens, cells, ledger = _chunk("b", 10)
    store.write_chunk(gens, cells, ledger)
    store.put_slot(RUN, 0, 0, "p0", 0, None, "initial")
    dest = tmp_path / "drive" / "snapshot.sqlite"
    assert store.backup_to(dest) == dest
    assert not Path(f"{dest}.tmp").exists() and not Path(f"{dest}-wal").exists()
    copy = Store(dest, read_only=True)
    assert copy.content_digest(RUN) == store.content_digest(RUN)
    assert copy.get_run(RUN)["config_hash"] == "cfg1"
    assert copy.quick_check()
    copy.close()
    # a second snapshot atomically replaces the first
    store.write_chunk(*_chunk("c", 2, draw=1))
    store.backup_to(dest)
    copy = Store(dest)
    assert copy.content_digest(RUN) == store.content_digest(RUN)
    copy.close()


def test_backup_refused_inside_transaction(store: Store, tmp_path: Path):
    with pytest.raises(StoreError), store.transaction():
        store.backup_to(tmp_path / "x.sqlite")


# ---------------------------------------------------------------- shards
def test_shard_writer_files_are_immutable_gzip_jsonl(store: Store, tmp_path: Path):
    shard_dir = tmp_path / "shards"
    w = ShardWriter.for_store(store, shard_dir)
    assert w.next_seq == 1
    store.write_chunk(*_chunk("s", 3), shard_writer=w, stage="matrix:seed/0")
    ((seq, path),) = list_shards(shard_dir)
    assert seq == 1 and path.name.startswith("000001_matrix-seed-0_") and path.name.endswith(".jsonl.gz")
    lines = [json.loads(x) for x in gzip.decompress(path.read_bytes()).decode().splitlines()]
    assert [x["table"] for x in lines] == ["generations"] * 3 + ["cells"] * 3 + ["ledger"]
    assert lines[-1]["row"]["id"] == 1  # autoincrement id travels with the row
    payload = read_shard(path)
    assert len(payload["generations"]) == 3
    assert store.get_meta("last_shard_seq") == "1"
    (log,) = store.shard_log()
    assert log["seq"] == 1 and log["path"] == path.name and log["n_rows"] == 7
    assert list(shard_dir.glob("*.tmp")) == []
    # identical payloads produce byte-identical files (gzip mtime is fixed)
    w2 = ShardWriter(tmp_path / "other", 1)
    _, p2, _, sha2 = w2.write("matrix:seed/0", payload)
    assert p2.name == path.name and sha2 == log["sha256"]


def test_shard_discarded_when_commit_fails(store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    w = ShardWriter(tmp_path / "shards", 1)

    def fail(*a, **k):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(store, "_record_shard", fail)
    with pytest.raises(sqlite3.OperationalError):
        store.write_chunk(*_chunk("f", 2), shard_writer=w, stage="x")
    assert list_shards(tmp_path / "shards") == [] and w.next_seq == 1
    assert store.count_rows("generations") == 0 and store.count_rows("ledger") == 0
    with pytest.raises(StoreError), store.transaction():
        store.write_chunk(*_chunk("f", 2), shard_writer=w, stage="x")


def test_shard_replay_restores_identical_store(store: Store, tmp_path: Path):
    shard_dir = tmp_path / "drive" / "shards"
    _ensure(store, synthetic=True)
    store.put_items("test", _items(3))
    w = ShardWriter.for_store(store, shard_dir)
    store.write_chunk(*_chunk("x", 3, draw=0), shard_writer=w, stage="matrix")
    store.put_slot(RUN, 0, 0, "p0", 0, None, "initial", shard_writer=w)
    snapshot = store.backup_to(tmp_path / "drive" / "snapshot.sqlite")  # mid-way snapshot
    store.write_chunk(*_chunk("y", 3, draw=1), shard_writer=w, stage="matrix")
    store.put_slot(RUN, 0, 1, "p1", 1, 0, "proposer", shard_writer=w)
    store.put_trajectory_round(
        {
            "run_id": RUN,
            "seed": 0,
            "round": 1,
            "incumbent_slot": 0,
            "candidate_slot": 1,
            "inc_dev_acc": 0.5,
            "cand_dev_acc": 0.6,
            "advanced": 1,
            "n_attempts": 1,
            "is_fallback": 0,
        },
        shard_writer=w,
    )
    store.put_scores(
        [
            {
                "gen_key": "y000",
                "extractor": "v1",
                "ext_hash": "h",
                "extracted": "0",
                "method": "boxed",
                "gold": "0",
                "correct": 1,
            }
        ],
        shard_writer=w,
    )
    store.write_chunk(*_chunk("z", 2, draw=2), shard_writer=w, stage="matrix")
    assert store.get_meta("last_shard_seq") == "7"
    want = store.content_digest(RUN)

    # (1) snapshot taken mid-way + newer shards
    restored_path = tmp_path / "restore" / "driftlab.sqlite"
    restored_path.parent.mkdir()
    restored_path.write_bytes(snapshot.read_bytes())
    restored = Store(restored_path)
    assert restored.content_digest(RUN) != want
    assert restored.get_meta("last_shard_seq") == "2"
    n = replay_shards(restored, shard_dir)
    assert n == 7 + 2 + 1 + 1 + 5  # chunk y, slot p1 (+prompt), round, score, chunk z
    assert restored.content_digest(RUN) == want
    assert restored.get_meta("last_shard_seq") == "7"
    assert restored.get_run(RUN)["synthetic"] == 1
    assert [r["id"] for r in restored.ledger_rows(RUN)] == [r["id"] for r in store.ledger_rows(RUN)]
    assert replay_shards(restored, shard_dir) == 0  # idempotent
    # writing continues after the replayed shards
    assert ShardWriter.for_store(restored, shard_dir).next_seq == 8
    restored.close()

    # (2) a fresh store from shards alone (items are not sharded here, so restore them first)
    fresh = Store(tmp_path / "fresh.sqlite")
    replay_shards(fresh, shard_dir)
    assert fresh.content_digest(RUN) == want
    assert fresh.count_rows("ledger") == 3
    fresh.close()


def test_replay_detects_corrupt_shard(store: Store, tmp_path: Path):
    w = ShardWriter(tmp_path / "s", 1)
    store.write_chunk(*_chunk("c", 2), shard_writer=w, stage="m")
    ((_, path),) = list_shards(tmp_path / "s")
    data = bytearray(path.read_bytes())
    data[-5] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ShardCorrupt):
        replay_shards(Store(tmp_path / "fresh.sqlite"), tmp_path / "s")


def test_writer_quarantines_uncommitted_shards(store: Store, tmp_path: Path):
    d = tmp_path / "s"
    w = ShardWriter.for_store(store, d)
    store.write_chunk(*_chunk("q", 2), shard_writer=w, stage="m")
    # simulate a hard kill after the shard of seq 2 was published but before its commit
    ShardWriter(d, 2).write("m", {"generations": [_gen("never_committed")]})
    (d / "000003_m_deadbeef.jsonl.gz.tmp").write_bytes(b"partial")
    with pytest.warns(UserWarning, match="quarantined"):
        w2 = ShardWriter.for_store(store, d)
    assert w2.next_seq == 2 and len(w2.orphans) == 1 and w2.orphans[0].name.endswith(".orphan")
    assert [s for s, _ in list_shards(d)] == [1]
    assert list(d.glob("*.tmp")) == []
    fresh = Store(tmp_path / "fresh.sqlite")
    replay_shards(fresh, d)
    assert "never_committed" not in fresh.lookup_generations(["never_committed"])
    fresh.close()


def test_store_usable_from_threads(store: Store):
    errors: list[BaseException] = []

    def work(t: int) -> None:
        try:
            for j in range(20):
                store.write_chunk([_gen(f"t{t}_{j}")], [_cell(f"t{t}_{j}", j, seed=t)], _ledger(1, 1, seed=t))
                store.lookup_generations([f"t{t}_{j}"])
        except BaseException as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=work, args=(t,)) for t in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors
    assert store.count_rows("generations") == 80 and store.count_rows("ledger") == 80


def test_raw_connection_works_with_pandas(store: Store):
    import pandas as pd

    gens, cells, ledger = _chunk("p", 3)
    store.write_chunk(gens, cells, ledger)
    df = pd.read_sql_query(
        "SELECT item_idx, gen_key FROM cells WHERE run_id = ? ORDER BY item_idx", store.conn, params=(RUN,)
    )
    assert list(df["item_idx"]) == [0, 1, 2] and list(df["gen_key"]) == ["p000", "p001", "p002"]
    assert store.query("SELECT COUNT(*) AS n FROM generations") == [{"n": 3}]


def test_in_memory_store():
    s = Store(":memory:")
    s.put_items("test", _items(2))
    assert len(s.get_items("test")) == 2
    s.close()
