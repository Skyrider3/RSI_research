"""DB -> analysis objects: the correctness :class:`~driftlab.analysis.cube.Cube`, the trajectories and the
proposer attempts of one run (read-only; nothing is cached on disk).

:func:`load_cube` runs ONE SQL join of the run's cells (one split, proposer cells excluded) with their
generation (``finish_reason`` and a distinct-text id) and their score rows (every extractor, gold matching
``items.gold``), then fills the cube with vectorised numpy indexing (no Python loop over cells or items):

* axes: ``seeds`` sorted; ``decodings`` sorted with greedy-like ones (temperature 0 in the run's
  ``config_json``) first; ``extractors`` = registered extractor names present in the scores (natural order,
  ``v1`` before ``v2`` before ``v10``); ``draws`` sorted by kind (round < gt < audit) then index;
  ``n_slots`` = max slot + 1; ``n_items`` = number of items stored for the split;
* ``correct`` (-1 where a cell or a score is missing), ``gen_row`` (index into ``gen_keys``, sorted),
  ``text_row`` (dense ids of distinct response texts: equal ids <=> byte-identical texts), ``physical``
  (``cells.physical``), ``truncated`` (``finish_reason == "length"``), ``greedy_decodings`` (from the config)
  and ``synthetic`` (``runs.synthetic``).

:func:`load_trajectories` rebuilds each seed's :class:`~driftlab.analysis.cube.Trajectory` from ``slots`` and
``trajectory_rounds`` (the longest prefix of completed rounds) and validates it against the stored incumbents.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from driftlab.analysis.cube import DRAW_AUDIT, DRAW_GT, DRAW_ROUND, Cube, Trajectory, empty_cube
from driftlab.extraction import REGISTRY, extractor_hash
from driftlab.store.store import Store

DRAW_KIND_ORDER: dict[str, int] = {DRAW_ROUND: 0, DRAW_GT: 1, DRAW_AUDIT: 2}
_KIND_NAMES: tuple[str, ...] = tuple(sorted(DRAW_KIND_ORDER, key=DRAW_KIND_ORDER.__getitem__))
PROPOSER_DECODING = "proposer"

# Extractors are pivoted into columns (one LEFT JOIN on the scores primary key per registered extractor), so the
# join returns ONE row per cell and item. The text id is a dense rank over the response text computed inside
# SQLite, so no response text leaves the database (equal rank <=> byte-identical text, BINARY collation).
_TEXT_CTE = """
WITH g AS (
  SELECT gen_key, finish_reason, DENSE_RANK() OVER (ORDER BY response) - 1 AS text_id
  FROM generations
  WHERE gen_key IN (SELECT gen_key FROM cells WHERE run_id = ? AND split = ? AND decoding_id != ?)
)"""
_BASE_COLUMNS: tuple[str, ...] = (
    "seed",
    "decoding_id",
    "slot",
    "draw_kind",
    "draw",
    "item_idx",
    "gen_key",
    "physical",
    "finish_reason",
    "text_id",
)


def _cube_sql(n_extractors: int, window: bool) -> str:
    """The cube join for ``n_extractors`` pivoted score columns (``window=False``: no text ids in SQL)."""
    score_cols = "".join(f", s{x}.correct" for x in range(n_extractors))
    score_joins = "".join(
        f"\nLEFT JOIN scores s{x} ON s{x}.gen_key = c.gen_key AND s{x}.extractor = ? AND s{x}.gold = i.gold"
        for x in range(n_extractors)
    )
    head = _TEXT_CTE if window else ""
    text_col = "g.text_id" if window else "NULL"
    gen_join = (
        "LEFT JOIN g ON g.gen_key = c.gen_key"
        if window
        else "LEFT JOIN generations g ON g.gen_key = c.gen_key"
    )
    return (
        f"{head}\nSELECT c.seed, c.decoding_id, c.slot, c.draw_kind, c.draw, c.item_idx, c.gen_key, c.physical, "
        f"g.finish_reason, {text_col}{score_cols}\nFROM cells c\n"
        f"JOIN items i ON i.split = c.split AND i.idx = c.item_idx\n{gen_join}{score_joins}\n"
        "WHERE c.run_id = ? AND c.split = ? AND c.decoding_id != ?"
    )


class TrajectoryInconsistent(ValueError):
    """The stored trajectory tables contradict each other (e.g. a stored incumbent the advance flags do not
    imply, a completed round without its slot, or a gap in the completed rounds)."""


# --------------------------------------------------------------------------- run / config helpers


def _open(db_path: str | Path) -> Store:
    return Store(db_path, read_only=True)


def _run_row(store: Store, run_id: str | None) -> dict:
    row = store.get_run(run_id)
    if row is None:
        what = f"run {run_id!r}" if run_id is not None else "any run"
        raise LookupError(f"{store.path}: no {what} in the database")
    return row


def run_config(row: Mapping[str, Any]) -> dict:
    """The run's stored config (``runs.config_json``) as a dict (``{}`` if absent or unparsable)."""
    raw = row.get("config_json")
    if not raw:
        return {}
    try:
        cfg = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError):
        return {}
    return cfg if isinstance(cfg, dict) else {}


def greedy_decodings(config: Mapping[str, Any]) -> frozenset[str]:
    """Decoding ids with temperature 0 in a config dict (``{"greedy"}`` when the config lists none)."""
    decs = config.get("decodings") if isinstance(config, Mapping) else None
    out: set[str] = set()
    if isinstance(decs, Mapping):
        for name, d in decs.items():
            try:
                temp = float(d.get("temperature")) if isinstance(d, Mapping) else float("nan")
            except (TypeError, ValueError):
                temp = float("nan")
            if temp == 0.0:
                out.add(str(name))
    return frozenset(out) if out else frozenset({"greedy"})


def _natural_key(name: str) -> tuple:
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", str(name)))


def _fetch(store: Store, run_id: str, split: str, names: list[str]) -> tuple[list[tuple], bool]:
    """Rows of the cube join (plain tuples) and whether ``text_id`` was computed in SQL."""
    where = (run_id, split, PROPOSER_DECODING)
    cur = store.conn.cursor()
    cur.row_factory = None  # plain tuples: much faster than sqlite3.Row for ~10^5 rows
    try:
        try:
            cur.execute(_cube_sql(len(names), True), (*where, *names, *where))
            return cur.fetchall(), True
        except sqlite3.OperationalError:  # SQLite < 3.25 has no window functions
            cur.execute(_cube_sql(len(names), False), (*names, *where))
            return cur.fetchall(), False
    finally:
        cur.close()


def _text_ids_python(store: Store, gen_keys: list[str]) -> dict[str, int]:
    """Fallback distinct-text ids by sha256 of the response (only without SQL window functions)."""
    ids: dict[str, int] = {}
    out: dict[str, int] = {}
    cur = store.conn.cursor()
    cur.row_factory = None
    try:
        for i in range(0, len(gen_keys), 900):
            part = gen_keys[i : i + 900]
            cur.execute(
                f"SELECT gen_key, response FROM generations WHERE gen_key IN ({','.join('?' * len(part))})",
                part,
            )
            for gk, resp in sorted(cur.fetchall()):
                h = hashlib.sha256(str(resp).encode("utf-8")).hexdigest()
                out[gk] = ids.setdefault(h, len(ids))
    finally:
        cur.close()
    return out


# --------------------------------------------------------------------------- cube


def load_cube(db_path: str | Path, run_id: str | None = None, split: str = "test") -> Cube:
    """Load the correctness cube of one run and split (see the module docstring).

    ``run_id=None`` selects the database's only run (``LookupError`` if there are none or several).
    """
    names = sorted(REGISTRY, key=_natural_key)
    with _open(db_path) as store:
        run = _run_row(store, run_id)
        rid = str(run["run_id"])
        greedy = greedy_decodings(run_config(run))
        item_idx = np.array(
            [r["idx"] for r in store.query("SELECT idx FROM items WHERE split = ? ORDER BY idx", (split,))],
            dtype=np.int64,
        )
        rows, sql_text_ids = _fetch(store, rid, split, names)
        if not rows:
            cube = empty_cube(rid, [], [], [], 0, [], int(len(item_idx)), greedy_decodings=sorted(greedy))
            cube.text_row = np.full(cube.gen_row.shape, -1, dtype=np.int32)
            cube.synthetic = bool(run["synthetic"])
            return cube
        cols = dict(zip((*_BASE_COLUMNS, *names), zip(*rows, strict=True), strict=True))
        gk_codes, gk_uniques = pd.factorize(np.asarray(cols["gen_key"], dtype=object), sort=True)
        gen_keys = [str(k) for k in gk_uniques]
        if sql_text_ids:
            text_id = np.asarray(cols["text_id"], dtype=float)
        else:
            tid = _text_ids_python(store, gen_keys)
            text_id = np.array([tid.get(k, -1) for k in gen_keys], dtype=float)[gk_codes]
    return _build_cube(
        cols, names, gk_codes, gen_keys, text_id, rid, item_idx, greedy, bool(run["synthetic"])
    )


def _build_cube(
    cols: Mapping[str, tuple],
    names: list[str],
    gk_codes: np.ndarray,
    gen_keys: list[str],
    text_id: np.ndarray,
    rid: str,
    item_idx: np.ndarray,
    greedy: frozenset[str],
    synthetic: bool,
) -> Cube:
    seed_col = np.asarray(cols["seed"], dtype=np.int64)
    seeds = [int(s) for s in np.unique(seed_col)]
    s_idx = np.searchsorted(np.asarray(seeds, dtype=np.int64), seed_col)

    dec_codes, dec_uniques = pd.factorize(np.asarray(cols["decoding_id"], dtype=object))
    dec_names = [str(d) for d in dec_uniques]
    decodings = sorted(dec_names, key=lambda d: (d not in greedy, d))
    d_idx = np.array([decodings.index(d) for d in dec_names], dtype=np.int64)[dec_codes]

    kind_codes, kind_uniques = pd.factorize(np.asarray(cols["draw_kind"], dtype=object))
    unknown = [str(k) for k in kind_uniques if k not in DRAW_KIND_ORDER]
    if unknown:
        raise ValueError(f"unknown draw kinds in cells: {sorted(unknown)}")
    kind_order = np.array([DRAW_KIND_ORDER[k] for k in kind_uniques], dtype=np.int64)[kind_codes]
    pairs = np.stack([kind_order, np.asarray(cols["draw"], dtype=np.int64)], axis=1)
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
    r_idx = np.asarray(inv).reshape(-1)
    draws = [(_KIND_NAMES[int(k)], int(d)) for k, d in uniq]

    slot_col = np.asarray(cols["slot"], dtype=np.int64)
    if (slot_col < 0).any():
        raise ValueError("cells with a negative slot")
    n_pos = np.searchsorted(item_idx, np.asarray(cols["item_idx"], dtype=np.int64))

    scores = {x: np.asarray(cols[x], dtype=float) for x in names}  # NaN = no score row
    extractors = [x for x in names if np.isfinite(scores[x]).any()]

    cube = empty_cube(
        rid,
        seeds,
        decodings,
        extractors,
        int(slot_col.max()) + 1,
        draws,
        int(len(item_idx)),
        greedy_decodings=sorted(greedy),
    )
    cube.gen_keys = gen_keys
    cell = (s_idx, d_idx, slot_col, r_idx)
    at = (*cell, n_pos)
    cube.gen_row[at] = np.asarray(gk_codes, dtype=np.int32)
    phys = np.asarray([bool(v) for v in cols["physical"]], dtype=bool)
    cube.physical[tuple(a[phys] for a in cell)] = True
    cube.truncated[at] = np.asarray(cols["finish_reason"], dtype=object) == "length"
    text_row = np.full(cube.gen_row.shape, -1, dtype=np.int32)
    text_row[at] = np.where(np.isfinite(text_id), text_id, -1).astype(np.int32)
    cube.text_row = text_row
    for xi, x in enumerate(extractors):
        v = scores[x]
        ok = np.isfinite(v)
        cube.correct[s_idx[ok], d_idx[ok], slot_col[ok], r_idx[ok], xi, n_pos[ok]] = (v[ok] != 0).astype(
            np.int8
        )
    cube.synthetic = synthetic
    return cube


def score_status(db_path: str | Path, run_id: str | None = None, split: str = "test") -> pd.DataFrame:
    """Score rows of the run's cells per (extractor, ext_hash): columns ``extractor, ext_hash, current_hash,
    n, stale`` (``stale`` = scored by an extractor source other than the current one, or unregistered)."""
    with _open(db_path) as store:
        rid = str(_run_row(store, run_id)["run_id"])
        rows = store.query(
            "SELECT s.extractor AS extractor, s.ext_hash AS ext_hash, COUNT(*) AS n FROM scores s "
            "WHERE s.gen_key IN (SELECT gen_key FROM cells WHERE run_id = ? AND split = ? AND decoding_id != ?) "
            "GROUP BY s.extractor, s.ext_hash ORDER BY s.extractor, s.ext_hash",
            (rid, split, PROPOSER_DECODING),
        )
    out = []
    for r in rows:
        cur = extractor_hash(r["extractor"]) if r["extractor"] in REGISTRY else None
        out.append({**r, "current_hash": cur, "stale": cur is None or cur != r["ext_hash"]})
    cols = ["extractor", "ext_hash", "current_hash", "n", "stale"]
    return (
        pd.DataFrame(out, columns=cols) if out else pd.DataFrame({c: pd.Series(dtype=object) for c in cols})
    )


# --------------------------------------------------------------------------- trajectories


def _float(v: object) -> float:
    return math.nan if v is None else float(v)  # type: ignore[arg-type]


def load_trajectories(db_path: str | Path, run_id: str | None = None) -> dict[int, Trajectory]:
    """``{seed: Trajectory}`` rebuilt from ``slots`` and ``trajectory_rounds`` (no writes).

    Each trajectory covers round 0 plus the completed rounds 1..R_s (``R_s`` = last completed round; slots
    written for a round that has not committed yet are ignored). Semantics are exactly those of
    :class:`~driftlab.analysis.cube.Trajectory`: ``inc_slot[0] = inc_slot[1] = 0``,
    ``inc_slot[t+1] = t if advanced[t] else inc_slot[t]``; ``cand_dev_acc[0]`` is slot 0's dev accuracy (round 1's
    ``inc_dev_acc``) and ``inc_dev_acc[0]`` is NaN. Seeds without slot 0 are omitted. Raises
    :class:`TrajectoryInconsistent` when a stored ``incumbent_slot`` / ``candidate_slot`` disagrees with the
    reconstruction, a completed round lacks its slot or prompt text, or completed rounds have a gap.
    """
    with _open(db_path) as store:
        rid = str(_run_row(store, run_id)["run_id"])
        slots = store.get_slots(rid)
        rounds = store.get_trajectory_rounds(rid)
    by_slot: dict[int, dict[int, dict]] = {}
    for r in slots:
        by_slot.setdefault(int(r["seed"]), {})[int(r["slot"])] = r
    by_round: dict[int, dict[int, dict]] = {}
    for r in rounds:
        by_round.setdefault(int(r["seed"]), {})[int(r["round"])] = r
    out: dict[int, Trajectory] = {}
    for seed in sorted(set(by_slot) | set(by_round)):
        sl, rd = by_slot.get(seed, {}), by_round.get(seed, {})
        if 0 not in sl:
            if rd:
                raise TrajectoryInconsistent(f"run {rid!r} seed {seed}: completed rounds but no slot 0")
            continue
        R = max(rd, default=0)
        missing = [t for t in range(1, R + 1) if t not in rd]
        if missing:
            raise TrajectoryInconsistent(
                f"run {rid!r} seed {seed}: rounds {missing} are not complete but round {R} is"
            )
        out[seed] = _trajectory(rid, seed, sl, rd, R)
    return out


def _trajectory(rid: str, seed: int, sl: Mapping[int, dict], rd: Mapping[int, dict], R: int) -> Trajectory:
    where = f"run {rid!r} seed {seed}"
    prompts, hashes, created, origin = [], [], [], []
    for k in range(R + 1):
        row = sl.get(k)
        if row is None or row.get("prompt_text") is None:
            raise TrajectoryInconsistent(
                f"{where}: round {k} is complete but slot {k} (or its prompt) is missing"
            )
        if int(row["created_round"]) != k:
            raise TrajectoryInconsistent(f"{where}: slot {k} has created_round {row['created_round']}")
        prompts.append(str(row["prompt_text"]))
        hashes.append(str(row["prompt_hash"]))
        created.append(k)
        origin.append(str(row["origin"]))
    advanced = [False] + [bool(rd[t]["advanced"]) for t in range(1, R + 1)]
    inc = [0] * (R + 1)
    for t in range(1, R + 1):
        inc[t] = 0 if t == 1 else ((t - 1) if advanced[t - 1] else inc[t - 1])
        stored_inc, stored_cand = int(rd[t]["incumbent_slot"]), int(rd[t]["candidate_slot"])
        if stored_inc != inc[t] or stored_cand != t:
            raise TrajectoryInconsistent(
                f"{where} round {t}: stored incumbent/candidate {stored_inc}/{stored_cand}, the advance flags "
                f"imply {inc[t]}/{t}"
            )
    acc0 = _float(rd[1]["inc_dev_acc"]) if R >= 1 else math.nan
    return Trajectory(
        seed=int(seed),
        prompts=prompts,
        prompt_hashes=hashes,
        created_round=created,
        inc_slot=inc,
        advanced=advanced,
        inc_dev_acc=[math.nan] + [_float(rd[t]["inc_dev_acc"]) for t in range(1, R + 1)],
        cand_dev_acc=[acc0] + [_float(rd[t]["cand_dev_acc"]) for t in range(1, R + 1)],
        is_fallback=[False] + [bool(rd[t]["is_fallback"]) for t in range(1, R + 1)],
        origin=origin,
    )


def proposer_attempts(db_path: str | Path, run_id: str | None = None) -> dict[int, dict[int, int]]:
    """``{seed: {round: n_attempts}}``: ``trajectory_rounds.n_attempts`` for completed rounds, else the number of
    stored ``proposals`` rows (an interrupted round's attempts)."""
    with _open(db_path) as store:
        rid = str(_run_row(store, run_id)["run_id"])
        done = store.query("SELECT seed, round, n_attempts FROM trajectory_rounds WHERE run_id = ?", (rid,))
        props = store.query(
            "SELECT seed, round, COUNT(*) AS n FROM proposals WHERE run_id = ? GROUP BY seed, round", (rid,)
        )
    out: dict[int, dict[int, int]] = {}
    for r in props:
        out.setdefault(int(r["seed"]), {})[int(r["round"])] = int(r["n"])
    for r in done:
        out.setdefault(int(r["seed"]), {})[int(r["round"])] = int(r["n_attempts"])
    return {s: dict(sorted(v.items())) for s, v in sorted(out.items())}


__all__ = [
    "DRAW_KIND_ORDER",
    "TrajectoryInconsistent",
    "greedy_decodings",
    "load_cube",
    "load_trajectories",
    "proposer_attempts",
    "run_config",
    "score_status",
]
