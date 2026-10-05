"""Scoring: every registered extractor over every referenced generation, idempotent, batched, no model calls."""

from __future__ import annotations

from pathlib import Path

import pytest

from driftlab.backends import make_backend
from driftlab.backends.base import GenRequest
from driftlab.backends.mock import MockBackend
from driftlab.config import REPO_ROOT, load_config
from driftlab.data import answer_key, load_dev, load_eval, user_message
from driftlab.engine import CellKey, CellTask, GenerationEngine
from driftlab.extraction import METHODS, REGISTRY, extract, extractor_hash, is_correct
from driftlab.keys import sample_seed
from driftlab.scoring import score_all, score_row, scoring_progress
from driftlab.store import ShardWriter, Store, replay_shards

CFG = load_config(REPO_ROOT / "configs" / "smoke_mock.yaml")
RUN = CFG.run.name
PROMPTS = (
    "You are a helpful assistant. Solve step by step and put the final answer within \\boxed{}.",
    "Answer as briefly as possible.",
)


def _populate(path: Path, shard_dir: Path | None = None) -> Store:
    """Items + greedy/t02 cells on both splits (seed 1 re-uses seed 0's greedy generations)."""
    store = Store(path)
    sw = ShardWriter.for_store(store, shard_dir) if shard_dir is not None else None
    dev, ev = load_dev(CFG), load_eval(CFG)
    store.put_items("train", [it.to_row() for it in dev], shard_writer=sw)
    store.put_items("test", [it.to_row() for it in ev], shard_writer=sw)
    backend = make_backend(CFG, answer_key=answer_key(CFG))
    engine = GenerationEngine(backend, store, RUN, chunk_size=10, shard_writer=sw)
    for seed in (0, 1):
        tasks = []
        for slot, prompt in enumerate(PROMPTS):
            for split, items in (("train", dev), ("test", ev)):
                for it in items:
                    for d in ("greedy", "t02"):
                        dec = CFG.decoding(d)
                        s = (
                            None
                            if dec.is_greedy
                            else sample_seed(seed, "eval", d, slot, "round", slot, split, it.idx)
                        )
                        tasks.append(
                            CellTask(
                                CellKey(RUN, seed, split, d, slot, "round", slot, it.idx),
                                GenRequest(prompt, user_message(CFG, it.question), dec, seed=s),
                                True,
                            )
                        )
        engine.run(tasks, stage="matrix", purpose="eval_matrix", seed=seed)
    return store


def _pairs(store: Store) -> set[tuple[str, str]]:
    rows = store.query(
        "SELECT DISTINCT c.gen_key, i.gold FROM cells c JOIN items i ON i.split = c.split AND i.idx = c.item_idx"
    )
    return {(r["gen_key"], r["gold"]) for r in rows}


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    s = _populate(tmp_path / "s.sqlite")
    yield s
    s.close()


def test_scores_every_generation_under_every_extractor(store: Store, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a, **k):  # scoring must never call a model
        raise AssertionError("model call during scoring")

    monkeypatch.setattr(MockBackend, "generate", boom)
    pairs = _pairs(store)
    n_cells = store.count_rows("cells")
    assert len(pairs) < n_cells  # greedy generations shared across seeds
    before = scoring_progress(store, RUN)
    assert before == {x: {"planned": len(pairs), "unscored": len(pairs)} for x in REGISTRY}
    counts = score_all(store, RUN)
    assert counts == {x: len(pairs) for x in REGISTRY}
    rows = store.query("SELECT * FROM scores")
    assert len(rows) == len(pairs) * len(REGISTRY)
    gens = store.lookup_generations({g for g, _ in pairs})
    for r in rows:
        assert (r["gen_key"], r["gold"]) in pairs
        assert r["ext_hash"] == extractor_hash(r["extractor"]) and r["method"] in METHODS
        ex = extract(r["extractor"], gens[r["gen_key"]]["response"])
        assert r["extracted"] == ex.extracted and r["correct"] == int(is_correct(ex, r["gold"]))
        if ex.span is None:
            assert r["span_start"] is None and r["span_end"] is None
        else:
            assert (r["span_start"], r["span_end"]) == tuple(ex.span)
    # v1-correct implies v2-correct (frozen extractor monotonicity) on real mock outputs
    by = {(r["gen_key"], r["gold"], r["extractor"]): r["correct"] for r in rows}
    assert all(by[(g, gold, "v2")] >= by[(g, gold, "v1")] for g, gold in pairs)
    assert scoring_progress(store, RUN) == {x: {"planned": len(pairs), "unscored": 0} for x in REGISTRY}
    assert score_all(store, RUN) == {x: 0 for x in REGISTRY}  # idempotent
    assert store.count_rows("scores") == len(rows)


def test_batching_is_invisible(tmp_path: Path, store: Store) -> None:
    other = _populate(tmp_path / "other.sqlite")
    assert score_all(store, batch=7) == score_all(other, batch=5000)
    assert store.content_digest(RUN) == other.content_digest(RUN)
    with pytest.raises(ValueError):
        score_all(store, batch=0)


def test_extractor_subset_and_unknown(store: Store) -> None:
    n = len(_pairs(store))
    assert score_all(store, extractors=["v1"]) == {"v1": n}
    assert {r["extractor"] for r in store.query("SELECT DISTINCT extractor FROM scores")} == {"v1"}
    assert scoring_progress(store, extractors=["v2"]) == {"v2": {"planned": n, "unscored": n}}
    with pytest.raises(KeyError, match="v9"):
        score_all(store, extractors=["v9"])


def test_stale_extractor_hash_is_rescored(store: Store) -> None:
    score_all(store)
    total = store.count_rows("scores")
    with store.transaction():
        store.conn.execute(
            "UPDATE scores SET ext_hash = 'stale', correct = 1 - correct WHERE extractor = 'v2'"
        )
    n = len(_pairs(store))
    assert scoring_progress(store)["v2"] == {"planned": n, "unscored": n}
    assert score_all(store) == {"v1": 0, "v2": n}
    assert store.count_rows("scores") == total
    stale = store.query("SELECT COUNT(*) AS n FROM scores WHERE ext_hash = 'stale'")[0]["n"]
    assert stale == 0
    gens = store.lookup_generations({g for g, _ in _pairs(store)})
    for r in store.query("SELECT * FROM scores WHERE extractor = 'v2'"):
        assert r["correct"] == int(is_correct(extract("v2", gens[r["gen_key"]]["response"]), r["gold"]))


def test_score_row_shape() -> None:
    row = score_row("k", "So the total is 3 + 4 = 7.\nThe final answer is \\boxed{7}.", "7", "v1", "h")
    assert row == {
        "gen_key": "k",
        "extractor": "v1",
        "ext_hash": "h",
        "extracted": "7",
        "method": "boxed",
        "span_start": row["span_start"],
        "span_end": row["span_end"],
        "gold": "7",
        "correct": 1,
    }
    assert row["span_start"] is not None and row["span_end"] > row["span_start"]
    miss = score_row("k", "no number here", "7", "v1", "h")
    assert miss["extracted"] is None and miss["correct"] == 0 and miss["span_start"] is None


def test_sharded_scores_replay(tmp_path: Path) -> None:
    shards = tmp_path / "shards"
    src = _populate(tmp_path / "src.sqlite", shard_dir=shards)
    score_all(src, shard_writer=ShardWriter.for_store(src, shards))
    fresh = Store(tmp_path / "fresh.sqlite")
    replay_shards(fresh, shards)
    assert fresh.content_digest(RUN) == src.content_digest(RUN)
    assert score_all(fresh) == {x: 0 for x in REGISTRY}
