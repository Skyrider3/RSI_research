"""Scoring: every registered extractor over every generation referenced by a cell (no model calls).

``score_all`` asks the store for generations that lack a ``scores`` row for the extractor's CURRENT source hash
(:meth:`Store.unscored`, gold taken from the cell's split / item) and writes one row per (generation, extractor,
gold). It is idempotent: a second call finds nothing to do; a changed extractor source (new ``ext_hash``)
re-scores everything under that extractor, replacing the stale rows.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from driftlab.extraction import REGISTRY, extract, extractor_hash, is_correct
from driftlab.store.shards import ShardWriter
from driftlab.store.store import Store

STAGE = "score"


def score_row(gen_key: str, response: str, gold: str, extractor: str, ext_hash: str) -> dict[str, Any]:
    """One ``scores`` row for a response under ``extractor``."""
    ex = extract(extractor, response)
    return {
        "gen_key": gen_key,
        "extractor": extractor,
        "ext_hash": ext_hash,
        "extracted": ex.extracted,
        "method": ex.method,
        "span_start": None if ex.span is None else int(ex.span[0]),
        "span_end": None if ex.span is None else int(ex.span[1]),
        "gold": gold,
        "correct": int(is_correct(ex, gold)),
    }


def _names(extractors: Iterable[str] | None) -> list[str]:
    names = list(REGISTRY) if extractors is None else list(dict.fromkeys(extractors))
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise KeyError(f"unknown extractor(s) {unknown}; registered: {sorted(REGISTRY)}")
    return names


def score_all(
    store: Store,
    run_id: str | None = None,
    *,
    extractors: Iterable[str] | None = None,
    batch: int = 5000,
    shard_writer: ShardWriter | None = None,
) -> dict[str, int]:
    """Score every unscored generation under each extractor (default: all registered); returns
    ``{extractor: n_rows_written}``. ``run_id`` is accepted for symmetry: the store scores every generation
    referenced by any cell (scores are keyed by generation, not by run)."""
    del run_id
    if int(batch) < 1:
        raise ValueError("batch must be >= 1")
    out: dict[str, int] = {}
    for name in _names(extractors):
        h = extractor_hash(name)
        n = 0
        prev: tuple[str, str] | None = None
        while True:
            rows = store.unscored(name, h, limit=int(batch))
            if not rows:
                break
            first = (rows[0]["gen_key"], rows[0]["gold"])
            if first == prev:  # defensive: the store did not accept the rows (would loop forever)
                raise RuntimeError(f"scores for {first} under {name}@{h} were not persisted")
            prev = first
            store.put_scores(
                [score_row(r["gen_key"], r["response"], r["gold"], name, h) for r in rows],
                shard_writer=shard_writer,
                stage=STAGE,
            )
            n += len(rows)
        out[name] = n
    return out


def scoring_progress(
    store: Store, run_id: str | None = None, *, extractors: Iterable[str] | None = None
) -> dict[str, dict[str, int]]:
    """``{extractor: {"planned", "unscored"}}`` over the distinct (generation, gold) pairs referenced by the
    run's cells (all runs if ``run_id`` is None) whose generation is stored (one pass for all extractors)."""
    names = _names(extractors)
    if not names:
        return {}
    where, run_params = ("WHERE c.run_id = ?", [run_id]) if run_id is not None else ("", [])
    sums = ", ".join(
        f"COALESCE(SUM(CASE WHEN NOT EXISTS (SELECT 1 FROM scores s WHERE s.gen_key = u.gen_key "
        f"AND s.extractor = ? AND s.gold = u.gold AND s.ext_hash = ?) THEN 1 ELSE 0 END), 0) AS u{i}"
        for i in range(len(names))
    )
    params = [v for name in names for v in (name, extractor_hash(name))]
    row = store.query(
        f"SELECT COUNT(*) AS planned, {sums} FROM (SELECT DISTINCT c.gen_key AS gen_key, i.gold AS gold "
        f"FROM cells c JOIN items i ON i.split = c.split AND i.idx = c.item_idx {where}) u "
        "JOIN generations g ON g.gen_key = u.gen_key",
        (*params, *run_params),
    )[0]
    return {
        name: {"planned": int(row["planned"]), "unscored": int(row[f"u{i}"])} for i, name in enumerate(names)
    }


__all__ = ["STAGE", "score_all", "score_row", "scoring_progress"]
