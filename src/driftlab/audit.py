"""Determinism audit (docs/ARCHITECTURE.md section 4): physical regenerations of creation cells.

For ``audit.seed``, slots {0, last trajectory incumbent} (deduplicated), each decoding in ``audit.decodings`` and
repeat ``a < audit.repeats``: the first ``audit.n_items`` eval items are regenerated as cells
``(k, ("audit", a))`` with nonce ``audit:s{seed}:{a}`` (forces a fresh generation for BOTH decodings); sampling
requests reuse the creation cell's sampling seed, so they test per-request seed reproducibility. Requests are
shuffled (``random.Random(rng_seed("audit", seed))``) and chunked differently from the main matrix, so batch
composition and order differ. Each regeneration is compared with the creation cell ``(k, ("round", k))``:
``pct_text_identical`` (byte-identical response) and ``pct_correct_flip`` (correctness differs) under the strict
extractor v1 -> ``audit_results`` (written through the engine's shard writer, so a snapshot + shard restore
recovers them); the returned dicts also carry the v2 flip rate. Ledger purpose ``audit``.

Missing creation cells (audit run before the matrix) are generated FIRST, exactly as the matrix plans them, so
every audit draw is a later regeneration of an existing creation generation.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence

from driftlab import keys
from driftlab.analysis.cube import DRAW_ROUND, Trajectory
from driftlab.config import ExperimentConfig
from driftlab.data import EVAL_SPLIT, EvalSplit
from driftlab.engine import GenerationEngine
from driftlab.extraction import extract, is_correct
from driftlab.planning import PURPOSE_MATRIX, audit_items, audit_slots, audit_tasks, creation_tasks
from driftlab.store.store import Store

STAGE = "audit"
PURPOSE = "audit"
STRICT_EXTRACTOR = "v1"
LENIENT_EXTRACTOR = "v2"
AUDIT_COLUMNS = ("run_id", "decoding_id", "slot", "repeat", "pct_text_identical", "pct_correct_flip", "n")


def _pct(count: int, n: int) -> float:
    return 100.0 * count / n if n else float("nan")


def audit_chunk_size(chunk_size: int) -> int:
    """A chunk size different from the matrix's (two thirds of it, at least 1)."""
    c = int(chunk_size)
    return max(1, (2 * c) // 3) if c > 1 else 1


def _creation_texts(
    cfg: ExperimentConfig,
    engine: GenerationEngine,
    store: Store,
    run_id: str,
    traj: Trajectory,
    eval_items: EvalSplit,
    decoding_id: str,
    slot: int,
    idxs: Sequence[int],
) -> dict[int, str]:
    """``{item_idx: text}`` of the creation cell; missing items are generated as the matrix would."""
    rows = store.get_cell_texts(run_id, traj.seed, EVAL_SPLIT, decoding_id, slot, DRAW_ROUND, slot)
    texts = {int(r["item_idx"]): r["response"] for r in rows}
    if all(i in texts for i in idxs):
        return texts
    tasks = creation_tasks(cfg, traj, eval_items, decoding_id, slot, run_id=run_id)
    recs = engine.run(tasks, stage="matrix", purpose=PURPOSE_MATRIX, seed=traj.seed, round_=slot)
    return {t.cell.item_idx: r.text for t, r in zip(tasks, recs, strict=True) if t.cell is not None}


def run_audit(
    cfg: ExperimentConfig,
    engine: GenerationEngine,
    store: Store,
    run_id: str,
    trajs: Mapping[int, Trajectory],
    eval_items: EvalSplit,
    *,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Run the audit, write ``audit_results`` and return one dict per (decoding, slot, repeat) with
    ``pct_text_identical``, ``pct_correct_flip`` (v1), ``pct_correct_flip_v2`` and the raw counts."""
    if not isinstance(eval_items, EvalSplit):
        raise TypeError(
            f"the audit uses the EVAL split: expected an EvalSplit, got {type(eval_items).__name__}"
        )
    seed = cfg.audit.seed
    tasks = audit_tasks(cfg, trajs, eval_items, run_id=run_id)
    if not tasks:
        return []
    traj = trajs[seed]
    items = audit_items(cfg, eval_items)
    idxs = [it.idx for it in items]
    gold = {it.idx: it.gold for it in items}
    slots = audit_slots(cfg, traj)
    base = {  # creation cells first: the audit regenerations are always later draws
        (d, k): _creation_texts(cfg, engine, store, run_id, traj, eval_items, d, k, idxs)
        for k in slots
        for d in cfg.audit.decodings
    }
    order = list(range(len(tasks)))
    random.Random(keys.rng_seed("audit", seed)).shuffle(order)
    shuffled = [tasks[i] for i in order]
    old = engine.chunk_size
    engine.chunk_size = audit_chunk_size(old)
    try:
        recs = engine.run(shuffled, stage=STAGE, purpose=PURPOSE, seed=seed)
    finally:
        engine.chunk_size = old
    audit_text: dict[tuple[str, int, int, int], str] = {}
    for t, r in zip(shuffled, recs, strict=True):
        assert t.cell is not None
        audit_text[(t.cell.decoding_id, t.cell.slot, t.cell.draw, t.cell.item_idx)] = r.text

    def correct(extractor: str, text: str, idx: int) -> bool:
        return is_correct(extract(extractor, text), gold[idx])

    rows: list[dict] = []
    for k in slots:
        for d in cfg.audit.decodings:
            for a in range(cfg.audit.repeats):
                n = ident = flip1 = flip2 = 0
                for i in idxs:
                    new, old_text = audit_text[(d, k, a, i)], base[(d, k)][i]
                    n += 1
                    ident += int(new == old_text)
                    flip1 += int(correct(STRICT_EXTRACTOR, new, i) != correct(STRICT_EXTRACTOR, old_text, i))
                    flip2 += int(
                        correct(LENIENT_EXTRACTOR, new, i) != correct(LENIENT_EXTRACTOR, old_text, i)
                    )
                rows.append(
                    {
                        "run_id": run_id,
                        "decoding_id": d,
                        "slot": k,
                        "repeat": a,
                        "pct_text_identical": _pct(ident, n),
                        "pct_correct_flip": _pct(flip1, n),
                        "n": n,
                        "seed": seed,
                        "pct_correct_flip_v1": _pct(flip1, n),
                        "pct_correct_flip_v2": _pct(flip2, n),
                        "n_text_identical": ident,
                        "n_flip_v1": flip1,
                        "n_flip_v2": flip2,
                    }
                )
    # Store.write_tables (not put_audit_results) so the rows also reach the shard log of a restorable run.
    store.write_tables(
        {"audit_results": [{c: r[c] for c in AUDIT_COLUMNS} for r in rows]},
        shard_writer=engine.shard_writer,
        stage=STAGE,
    )
    for r in rows:
        log(
            f"[audit] seed {seed} {r['decoding_id']} slot {r['slot']} repeat {r['repeat']}: "
            f"{r['pct_text_identical']:.1f}% identical text, {r['pct_correct_flip']:.1f}% v1 flips "
            f"({r['pct_correct_flip_v2']:.1f}% v2) over {r['n']} items"
        )
    return rows


__all__ = ["AUDIT_COLUMNS", "PURPOSE", "STAGE", "audit_chunk_size", "run_audit"]
