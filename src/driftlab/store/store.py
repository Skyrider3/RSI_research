"""SQLite experiment store (WAL mode) over the frozen schema in ``store/schema.sql``.

Write semantics per table (also used when shards are replayed, see ``store/shards.py``):

* content-addressed / logical tables (``generations``, ``cells``, ``items``, ``prompts``, ``slots``,
  ``ledger``, ...) are insert-or-ignore on key conflicts (``ON CONFLICT DO NOTHING``, so NOT NULL / CHECK
  violations still raise); ``slots`` additionally refuse a different ``prompt_hash`` for an existing slot
  (determinism guard, :class:`SlotConflict`);
* re-computable tables (``proposals``, ``trajectory_rounds``, ``stage_status``, ``audit_results``,
  ``analysis_results``, ``meta``) use ``INSERT OR REPLACE`` so idempotent re-runs converge;
* ``scores`` are inserted once per ``(gen_key, extractor, gold)`` and only overwritten when the frozen
  extractor's source hash (``ext_hash``) differs, so :meth:`Store.unscored` always terminates.

Timestamps (``created_at`` ...) are ISO-8601 UTC strings; they never enter hashes or digests.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import cache
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

from driftlab.keys import sha256_text

if TYPE_CHECKING:
    from driftlab.store.shards import ShardWriter

SCHEMA_VERSION = "1"
MAX_SQL_VARS = 900  # SQLite's historical limit is 999 bound variables per statement

REPLACE_TABLES = frozenset(
    {"proposals", "trajectory_rounds", "stage_status", "audit_results", "analysis_results", "meta"}
)
AUTOINC_TABLES = frozenset({"ledger", "interactive_events"})
NON_REPLAYABLE_TABLES = frozenset({"shard_log"})
TIMESTAMP_COLUMNS: dict[str, str] = {
    "runs": "created_at",
    "generations": "created_at",
    "ledger": "created_at",
    "analysis_results": "created_at",
    "interactive_events": "created_at",
    "trajectory_rounds": "completed_at",
    "stage_status": "updated_at",
}
CELL_KEY_COLUMNS: tuple[str, ...] = ("seed", "split", "decoding_id", "slot", "draw_kind", "draw", "item_idx")
CELL_COLUMNS: tuple[str, ...] = ("run_id", *CELL_KEY_COLUMNS, "gen_key", "physical")


class StoreError(RuntimeError):
    """Base class for store integrity errors."""


class ConfigMismatch(StoreError):
    """The run already exists with a different config hash."""


class EngineMismatch(StoreError):
    """The run already exists with a different engine fingerprint (e.g. resumed on another GPU)."""


class SlotConflict(StoreError):
    """A slot already exists with a different prompt (the trajectory is not deterministic)."""


def utc_now() -> str:
    """Current time as an ISO-8601 UTC string (for ``created_at`` columns only)."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@cache
def schema_sql() -> str:
    return resources.files("driftlab.store").joinpath("schema.sql").read_text(encoding="utf-8")


def _json_default(o: object) -> object:
    if hasattr(o, "tolist"):  # numpy arrays and scalars
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def to_json(obj: object) -> str:
    """Canonical JSON used for JSON-valued columns (sorted keys, compact, numpy-safe)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_json_default)


_PLAIN = (str, int, float, bytes, type(None))


def _coerce(v: Any) -> Any:
    """Make a value bindable by sqlite3 and JSON-serializable (lists/dicts -> JSON text)."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, _PLAIN):
        return v
    if isinstance(v, (list, tuple, dict, set, frozenset)):
        return to_json(v)
    if hasattr(v, "item"):  # numpy scalar
        return _coerce(v.item())
    if isinstance(v, Path):
        return str(v)
    raise TypeError(f"cannot store value of type {type(v).__name__}: {v!r}")


def _as_json_text(v: object) -> str | None:
    if v is None or isinstance(v, str):
        return v
    return to_json(v)


class Store:
    """Experiment database. One connection, usable from several threads (serialized by a lock)."""

    def __init__(self, path: str | Path, read_only: bool = False) -> None:
        self._path = Path(path)
        self.read_only = bool(read_only)
        self._lock = threading.RLock()
        self._depth = 0
        self._columns: dict[str, tuple[str, ...]] = {}
        memory = str(path) == ":memory:"
        if self.read_only:
            if memory or not self._path.exists():
                raise FileNotFoundError(f"no database at {self._path}")
            uri = self._path.resolve().as_uri() + "?mode=ro"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False, isolation_level=None, timeout=30.0)
        else:
            if not memory:
                self._path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None, timeout=30.0)
        conn.row_factory = sqlite3.Row
        self._conn = conn
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            if self.read_only:
                conn.execute("PRAGMA query_only=1")
                return
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(schema_sql())
            version = self.get_meta("schema_version")
            if version is None:
                self.set_meta("schema_version", SCHEMA_VERSION)
            elif version != SCHEMA_VERSION:
                raise StoreError(f"{self._path}: schema_version {version!r}, expected {SCHEMA_VERSION!r}")
        except BaseException:
            conn.close()
            raise

    # ------------------------------------------------------------------ connection helpers
    @property
    def path(self) -> Path:
        return self._path

    @property
    def conn(self) -> sqlite3.Connection:
        """Raw connection (rows are ``sqlite3.Row``), e.g. for ``pandas.read_sql_query``."""
        return self._conn

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[Store]:
        """``BEGIN IMMEDIATE`` ... ``COMMIT`` (rollback on error). Nested calls join the outer one."""
        with self._lock:
            if self._depth > 0:
                self._depth += 1
                try:
                    yield self
                finally:
                    self._depth -= 1
                return
            self._conn.execute("BEGIN IMMEDIATE")
            self._depth = 1
            try:
                yield self
            except BaseException:
                self._depth = 0
                self._conn.execute("ROLLBACK")
                raise
            self._depth = 0
            try:
                self._conn.execute("COMMIT")
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise

    @property
    def in_transaction(self) -> bool:
        return self._depth > 0

    def query(self, sql: str, params: Sequence[Any] | Mapping[str, Any] = ()) -> list[dict]:
        """Run a read query and return rows as dicts."""
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params)]

    def _one(self, sql: str, params: Sequence[Any] = ()) -> dict | None:
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return dict(row) if row is not None else None

    def _batched(self, sql: str, keys: Iterable[str], extra: Sequence[Any] = ()) -> list[dict]:
        """Run ``sql`` (containing ``{ph}`` for the IN list) over ``keys`` in batches of MAX_SQL_VARS."""
        uniq = list(dict.fromkeys(keys))
        out: list[dict] = []
        for i in range(0, len(uniq), MAX_SQL_VARS):
            part = uniq[i : i + MAX_SQL_VARS]
            out.extend(self.query(sql.format(ph=",".join("?" * len(part))), (*part, *extra)))
        return out

    def table_columns(self, table: str) -> tuple[str, ...]:
        """Column names of a schema table (raises ``ValueError`` for unknown tables)."""
        cols = self._columns.get(table)
        if cols is None:
            with self._lock:
                info = (
                    self._conn.execute(f"PRAGMA table_info({table})").fetchall()
                    if table.isidentifier()
                    else []
                )
            if not info:
                raise ValueError(f"unknown table {table!r}")
            cols = tuple(r["name"] for r in info)
            self._columns[table] = cols
        return cols

    def count_rows(self, table: str) -> int:
        self.table_columns(table)  # validates the name
        with self._lock:
            return int(self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    def quick_check(self) -> bool:
        with self._lock:
            return self._conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"

    # ------------------------------------------------------------------ generic row writing
    def _insert_sql(self, table: str, cols: tuple[str, ...]) -> str:
        names = ",".join(f'"{c}"' for c in cols)
        marks = ",".join("?" * len(cols))
        if table in REPLACE_TABLES:
            return f"INSERT OR REPLACE INTO {table} ({names}) VALUES ({marks})"
        upd = [c for c in cols if c not in ("gen_key", "extractor", "gold")]
        if table == "scores" and upd:
            sets = ",".join(f'"{c}"=excluded."{c}"' for c in upd)
            return (
                f"INSERT INTO scores ({names}) VALUES ({marks}) ON CONFLICT(gen_key, extractor, gold) "
                f"DO UPDATE SET {sets} WHERE scores.ext_hash IS NOT excluded.ext_hash"
            )
        # "INSERT OR IGNORE" semantics for key conflicts only: unlike OR IGNORE, ON CONFLICT DO NOTHING
        # still raises on NOT NULL / CHECK violations instead of silently dropping the row.
        return f"INSERT INTO {table} ({names}) VALUES ({marks}) ON CONFLICT DO NOTHING"

    def _insert_rows(self, table: str, rows: Iterable[Mapping[str, Any]]) -> list[dict]:
        """Insert rows with the table's conflict policy; returns the coerced rows (with assigned ids)."""
        allowed = set(self.table_columns(table))
        ts_col = TIMESTAMP_COLUMNS.get(table)
        groups: dict[tuple[str, ...], list[tuple]] = {}
        out: list[dict] = []
        for row in rows:
            r = {k: _coerce(v) for k, v in row.items()}
            unknown = set(r) - allowed
            if unknown:
                raise ValueError(f"unknown column(s) for {table}: {sorted(unknown)}")
            if ts_col is not None and not r.get(ts_col):
                r[ts_col] = utc_now()
            if table in AUTOINC_TABLES and r.get("id") is None:
                r.pop("id", None)
                cols = tuple(r)
                cur = self._conn.execute(self._insert_sql(table, cols), tuple(r.values()))
                r["id"] = cur.lastrowid
            else:
                groups.setdefault(tuple(r), []).append(tuple(r.values()))
            out.append(r)
        for cols, values in groups.items():
            self._conn.executemany(self._insert_sql(table, cols), values)
        if table == "slots":
            self._check_slots(out)
        return out

    def _check_slots(self, rows: list[dict]) -> None:
        for r in rows:
            got = self._conn.execute(
                "SELECT prompt_hash FROM slots WHERE run_id=? AND seed=? AND slot=?",
                (r["run_id"], r["seed"], r["slot"]),
            ).fetchone()
            if got is not None and got[0] != r["prompt_hash"]:
                raise SlotConflict(
                    f"slot (run={r['run_id']}, seed={r['seed']}, slot={r['slot']}) already holds prompt "
                    f"{got[0][:12]}, refusing {str(r['prompt_hash'])[:12]} (non-deterministic trajectory?)"
                )

    def write_tables(
        self,
        payload: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        shard_writer: ShardWriter | None = None,
        stage: str = "",
    ) -> None:
        """Write rows for several tables in ONE transaction (+ one shard file if ``shard_writer``).

        The shard is written (tmp + ``os.replace``) before the transaction commits and is recorded in
        ``shard_log`` / ``meta.last_shard_seq`` inside it; if the commit fails the shard is discarded.
        """
        with self._lock:
            if shard_writer is not None and self._depth > 0:
                raise StoreError("a sharded write must not be nested inside an outer transaction")
            shard: tuple[int, Path, int, str] | None = None
            try:
                with self.transaction():
                    written = {t: self._insert_rows(t, rows) for t, rows in payload.items() if rows}
                    if shard_writer is not None and written:
                        shard = shard_writer.write(stage, written)
                        seq, path, n_rows, sha = shard
                        self._record_shard(seq, Path(path).name, n_rows, sha)
            except BaseException:
                if shard is not None and shard_writer is not None:
                    shard_writer.discard(shard[0], shard[1])
                raise

    def apply_shard(
        self, seq: int, name: str, n_rows: int, sha256: str, payload: Mapping[str, Sequence[Mapping]]
    ) -> None:
        """Replay one shard's rows (table policies as above) and record it, in one transaction."""
        with self._lock, self.transaction():
            for table, rows in payload.items():
                if table not in NON_REPLAYABLE_TABLES and rows:
                    self._insert_rows(table, rows)
            self._record_shard(seq, name, n_rows, sha256)

    def _record_shard(self, seq: int, name: str, n_rows: int, sha256: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO shard_log (seq, path, n_rows, sha256) VALUES (?, ?, ?, ?)",
            (int(seq), name, int(n_rows), sha256),
        )
        if int(seq) > int(self.get_meta("last_shard_seq") or 0):
            self.set_meta("last_shard_seq", str(int(seq)))

    def shard_log(self) -> list[dict]:
        return self.query("SELECT * FROM shard_log ORDER BY seq")

    # ------------------------------------------------------------------ meta
    def get_meta(self, key: str) -> str | None:
        row = self._one("SELECT value FROM meta WHERE key = ?", (key,))
        return None if row is None else row["value"]

    def set_meta(self, key: str, value: object) -> None:
        with self.transaction():
            self._conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (key, None if value is None else str(value)),
            )

    def meta_items(self, prefix: str = "") -> dict[str, str | None]:
        rows = self.query(
            "SELECT key, value FROM meta WHERE substr(key, 1, ?) = ? ORDER BY key", (len(prefix), prefix)
        )
        return {r["key"]: r["value"] for r in rows}

    def _record_event(self, prefix: str, payload: Mapping[str, Any]) -> str:
        """Append ``meta['<prefix>:<n>']`` (timestamp-free counter) holding a JSON payload."""
        n = self._conn.execute(
            "SELECT COUNT(*) FROM meta WHERE substr(key, 1, ?) = ?", (len(prefix) + 1, prefix + ":")
        ).fetchone()[0]
        key = f"{prefix}:{n}"
        self.set_meta(key, to_json(dict(payload)))
        return key

    # ------------------------------------------------------------------ runs
    def ensure_run(
        self,
        run_id: str,
        *,
        config_json: str | Mapping[str, Any],
        config_hash: str,
        plan_hash: str | None = None,
        provenance_json: str | Mapping[str, Any] | None = None,
        engine_fp: str | None = None,
        synthetic: bool,
        allow_config_change: bool = False,
        allow_engine_change: bool = False,
    ) -> dict:
        """Create the run row or validate a resume against it; returns the (updated) row.

        * different ``config_hash`` -> :class:`ConfigMismatch` unless ``allow_config_change`` (then the row
          is updated and ``meta['config_change:<n>']`` records old/new);
        * different non-null ``engine_fp`` -> :class:`EngineMismatch` unless ``allow_engine_change``
          (recorded as ``meta['engine_change:<n>']``);
        * ``synthetic`` is sticky (stored = stored OR new); ``plan_hash`` follows the latest value (a change
          is recorded as ``meta['plan_change:<n>']``); ``provenance_json`` is kept from the first call.
        """
        cfg = _as_json_text(config_json)
        prov = _as_json_text(provenance_json)
        with self.transaction():
            row = self.get_run(run_id)
            if row is None:
                self._conn.execute(
                    "INSERT INTO runs (run_id, config_json, config_hash, plan_hash, provenance_json, "
                    "engine_fp, synthetic, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (run_id, cfg, config_hash, plan_hash, prov, engine_fp, int(bool(synthetic)), utc_now()),
                )
                return self.get_run(run_id)  # type: ignore[return-value]
            updates: dict[str, Any] = {}
            if row["config_hash"] != config_hash:
                if not allow_config_change:
                    raise ConfigMismatch(
                        f"run {run_id!r} was created with config {row['config_hash']}, got {config_hash} "
                        "(use a new run dir, or allow_config_change)"
                    )
                self._record_event(
                    "config_change", {"run_id": run_id, "old": row["config_hash"], "new": config_hash}
                )
                updates.update(config_json=cfg, config_hash=config_hash)
            if engine_fp is not None and row["engine_fp"] != engine_fp:
                if row["engine_fp"] is not None:
                    if not allow_engine_change:
                        raise EngineMismatch(
                            f"run {run_id!r} was generated with engine {row['engine_fp']}, this engine is "
                            f"{engine_fp} (resume on the same GPU/stack, or allow_engine_change)"
                        )
                    self._record_event(
                        "engine_change", {"run_id": run_id, "old": row["engine_fp"], "new": engine_fp}
                    )
                updates["engine_fp"] = engine_fp
            if synthetic and not row["synthetic"]:
                updates["synthetic"] = 1
            if plan_hash is not None and plan_hash != row["plan_hash"]:
                if row["plan_hash"] is not None:
                    self._record_event(
                        "plan_change", {"run_id": run_id, "old": row["plan_hash"], "new": plan_hash}
                    )
                updates["plan_hash"] = plan_hash
            if prov is not None and row["provenance_json"] is None:
                updates["provenance_json"] = prov
            if updates:
                sets = ", ".join(f"{k} = ?" for k in updates)
                self._conn.execute(f"UPDATE runs SET {sets} WHERE run_id = ?", (*updates.values(), run_id))
            return self.get_run(run_id)  # type: ignore[return-value]

    def get_run(self, run_id: str | None = None) -> dict | None:
        """The run row; with ``run_id=None`` the only run (``None`` if empty, ``LookupError`` if several)."""
        if run_id is not None:
            return self._one("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        rows = self.list_runs()
        if len(rows) > 1:
            raise LookupError(f"{len(rows)} runs in {self._path}; pass run_id explicitly")
        return rows[0] if rows else None

    def list_runs(self) -> list[dict]:
        return self.query("SELECT * FROM runs ORDER BY created_at, run_id")

    # ------------------------------------------------------------------ items & prompts
    def put_items(self, split: str, rows: Iterable[Mapping[str, Any]]) -> None:
        """Insert dataset items (``idx, question, answer_text, gold``); idempotent."""
        with self.transaction():
            self._insert_rows("items", [{**dict(r), "split": split} for r in rows])

    def get_items(self, split: str) -> list[dict]:
        return self.query("SELECT * FROM items WHERE split = ? ORDER BY idx", (split,))

    def put_prompt(self, text: str) -> str:
        h = sha256_text(text)
        with self.transaction():
            self._insert_rows("prompts", [{"prompt_hash": h, "text": text}])
        return h

    def get_prompt(self, prompt_hash: str) -> str | None:
        row = self._one("SELECT text FROM prompts WHERE prompt_hash = ?", (prompt_hash,))
        return None if row is None else row["text"]

    def get_prompts(self, hashes: Iterable[str]) -> dict[str, str]:
        rows = self._batched("SELECT prompt_hash, text FROM prompts WHERE prompt_hash IN ({ph})", hashes)
        return {r["prompt_hash"]: r["text"] for r in rows}

    # ------------------------------------------------------------------ trajectory tables
    def put_slot(
        self,
        run_id: str,
        seed: int,
        slot: int,
        prompt_text: str,
        created_round: int,
        parent_slot: int | None,
        origin: str,
        *,
        shard_writer: ShardWriter | None = None,
        stage: str = "trajectory",
    ) -> str:
        """Insert a slot (+ its prompt); returns the prompt hash. Raises :class:`SlotConflict` if the slot
        already exists with a different prompt."""
        h = sha256_text(prompt_text)
        self.write_tables(
            {
                "prompts": [{"prompt_hash": h, "text": prompt_text}],
                "slots": [
                    {
                        "run_id": run_id,
                        "seed": seed,
                        "slot": slot,
                        "prompt_hash": h,
                        "created_round": created_round,
                        "parent_slot": parent_slot,
                        "origin": origin,
                    }
                ],
            },
            shard_writer=shard_writer,
            stage=stage,
        )
        return h

    def get_slots(self, run_id: str, seed: int | None = None) -> list[dict]:
        """Slot rows joined with their prompt text (column ``prompt_text``), ordered by seed, slot."""
        sql = (
            "SELECT s.*, p.text AS prompt_text FROM slots s "
            "LEFT JOIN prompts p ON p.prompt_hash = s.prompt_hash WHERE s.run_id = ?"
        )
        params: list[Any] = [run_id]
        if seed is not None:
            sql += " AND s.seed = ?"
            params.append(seed)
        return self.query(sql + " ORDER BY s.seed, s.slot", params)

    def put_proposal(
        self, row: Mapping[str, Any], *, shard_writer: ShardWriter | None = None, stage: str = "trajectory"
    ) -> None:
        """INSERT OR REPLACE one proposer attempt (list-valued columns are stored as JSON)."""
        self.write_tables({"proposals": [row]}, shard_writer=shard_writer, stage=stage)

    def get_proposals(self, run_id: str, seed: int | None = None, round: int | None = None) -> list[dict]:
        sql, params = "SELECT * FROM proposals WHERE run_id = ?", [run_id]
        if seed is not None:
            sql += " AND seed = ?"
            params.append(seed)
        if round is not None:
            sql += " AND round = ?"
            params.append(round)
        return self.query(sql + " ORDER BY seed, round, attempt", params)

    def put_trajectory_round(
        self, row: Mapping[str, Any], *, shard_writer: ShardWriter | None = None, stage: str = "trajectory"
    ) -> None:
        """INSERT OR REPLACE the row that marks a trajectory round complete (``completed_at`` defaults to now)."""
        self.write_tables({"trajectory_rounds": [row]}, shard_writer=shard_writer, stage=stage)

    def get_trajectory_rounds(self, run_id: str, seed: int | None = None) -> list[dict]:
        sql, params = "SELECT * FROM trajectory_rounds WHERE run_id = ?", [run_id]
        if seed is not None:
            sql += " AND seed = ?"
            params.append(seed)
        return self.query(sql + " ORDER BY seed, round", params)

    # ------------------------------------------------------------------ generations & cells
    def lookup_generations(self, gen_keys: Iterable[str]) -> dict[str, dict]:
        """Existing generation rows by key (batched IN queries of <= MAX_SQL_VARS keys)."""
        rows = self._batched("SELECT * FROM generations WHERE gen_key IN ({ph})", gen_keys)
        return {r["gen_key"]: r for r in rows}

    def write_chunk(
        self,
        generations: Sequence[Mapping[str, Any]],
        cells: Sequence[Mapping[str, Any]],
        ledger: Mapping[str, Any] | None,
        shard_writer: ShardWriter | None = None,
        stage: str = "",
    ) -> None:
        """Commit one engine chunk (generations + cells + ledger row [+ shard]) in ONE transaction."""
        payload: dict[str, Sequence[Mapping[str, Any]]] = {"generations": generations, "cells": cells}
        if ledger is not None:
            payload["ledger"] = [ledger]
        self.write_tables(payload, shard_writer=shard_writer, stage=stage)

    def existing_cell_keys(
        self, run_id: str, split: str | None = None, seed: int | None = None
    ) -> set[tuple]:
        """``{(seed, split, decoding_id, slot, draw_kind, draw, item_idx), ...}`` already written."""
        sql = f"SELECT {', '.join(CELL_KEY_COLUMNS)} FROM cells WHERE run_id = ?"
        params: list[Any] = [run_id]
        if split is not None:
            sql += " AND split = ?"
            params.append(split)
        if seed is not None:
            sql += " AND seed = ?"
            params.append(seed)
        with self._lock:
            return {tuple(r) for r in self._conn.execute(sql, params)}

    def get_cells(self, run_id: str, split: str | None = None, seed: int | None = None) -> list[dict]:
        """Cell rows (cell columns only), ordered by the cell key."""
        sql = f"SELECT {', '.join(CELL_COLUMNS)} FROM cells WHERE run_id = ?"
        params: list[Any] = [run_id]
        if split is not None:
            sql += " AND split = ?"
            params.append(split)
        if seed is not None:
            sql += " AND seed = ?"
            params.append(seed)
        return self.query(sql + f" ORDER BY {', '.join(CELL_KEY_COLUMNS)}", params)

    def get_cell_texts(
        self, run_id: str, seed: int, split: str, decoding_id: str, slot: int, draw_kind: str, draw: int
    ) -> list[dict]:
        """``[{item_idx, gen_key, response, finish_reason}, ...]`` of one cell (drill-down), by item."""
        return self.query(
            "SELECT c.item_idx, c.gen_key, g.response, g.finish_reason FROM cells c "
            "JOIN generations g ON g.gen_key = c.gen_key WHERE c.run_id = ? AND c.seed = ? AND c.split = ? "
            "AND c.decoding_id = ? AND c.slot = ? AND c.draw_kind = ? AND c.draw = ? ORDER BY c.item_idx",
            (run_id, seed, split, decoding_id, slot, draw_kind, draw),
        )

    # ------------------------------------------------------------------ scores
    def unscored(self, extractor: str, ext_hash: str, limit: int | None = None) -> list[dict]:
        """``[{gen_key, response, gold}, ...]`` for generations referenced by cells (gold from the cell's
        split/item) that lack a ``scores`` row for ``(gen_key, extractor, gold)`` with ``ext_hash``."""
        return self.query(
            "SELECT u.gen_key AS gen_key, g.response AS response, u.gold AS gold FROM "
            "(SELECT DISTINCT c.gen_key AS gen_key, i.gold AS gold FROM cells c "
            " JOIN items i ON i.split = c.split AND i.idx = c.item_idx) u "
            "JOIN generations g ON g.gen_key = u.gen_key "
            "WHERE NOT EXISTS (SELECT 1 FROM scores s WHERE s.gen_key = u.gen_key AND s.extractor = ? "
            " AND s.gold = u.gold AND s.ext_hash = ?) "
            "ORDER BY u.gen_key, u.gold LIMIT ?",
            (extractor, ext_hash, -1 if limit is None else int(limit)),
        )

    def put_scores(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        shard_writer: ShardWriter | None = None,
        stage: str = "score",
    ) -> None:
        """Insert score rows (no-op for an existing row with the same ``ext_hash``)."""
        self.write_tables({"scores": list(rows)}, shard_writer=shard_writer, stage=stage)

    def get_scores(self, gen_keys: Iterable[str], extractor: str | None = None) -> list[dict]:
        if extractor is None:
            return self._batched("SELECT * FROM scores WHERE gen_key IN ({ph})", gen_keys)
        return self._batched(
            "SELECT * FROM scores WHERE gen_key IN ({ph}) AND extractor = ?", gen_keys, (extractor,)
        )

    # ------------------------------------------------------------------ ledger & stage status
    def ledger_rows(self, run_id: str) -> list[dict]:
        return self.query("SELECT * FROM ledger WHERE run_id = ? ORDER BY id", (run_id,))

    def set_stage_status(
        self,
        run_id: str,
        stage: str,
        seed: int | None,
        n_planned: int | None,
        n_done: int | None,
        status: str | None,
    ) -> None:
        """Upsert a stage progress row (``seed=None`` is stored as -1 = all seeds)."""
        with self.transaction():
            self._insert_rows(
                "stage_status",
                [
                    {
                        "run_id": run_id,
                        "stage": stage,
                        "seed": -1 if seed is None else seed,
                        "n_planned": n_planned,
                        "n_done": n_done,
                        "status": status,
                    }
                ],
            )

    def get_stage_status(self, run_id: str) -> list[dict]:
        return self.query("SELECT * FROM stage_status WHERE run_id = ? ORDER BY stage, seed", (run_id,))

    # ------------------------------------------------------------------ audit / analysis / interactive
    def put_audit_results(self, rows: Iterable[Mapping[str, Any]]) -> None:
        with self.transaction():
            self._insert_rows("audit_results", list(rows))

    def get_audit_results(self, run_id: str) -> list[dict]:
        return self.query(
            "SELECT * FROM audit_results WHERE run_id = ? ORDER BY decoding_id, slot, repeat", (run_id,)
        )

    def put_analysis_result(self, analysis_id: str, name: str, payload: dict | list) -> None:
        with self.transaction():
            self._insert_rows(
                "analysis_results",
                [{"analysis_id": analysis_id, "name": name, "payload_json": to_json(payload)}],
            )

    def get_analysis_result(self, analysis_id: str, name: str) -> dict | list | None:
        row = self._one(
            "SELECT payload_json FROM analysis_results WHERE analysis_id = ? AND name = ?",
            (analysis_id, name),
        )
        return None if row is None else json.loads(row["payload_json"])

    def put_interactive_event(self, **fields: Any) -> int:
        """Log a dashboard event (``backend`` required); returns its id."""
        if not fields.get("backend"):
            raise ValueError("interactive events need a 'backend'")
        with self.transaction():
            (row,) = self._insert_rows("interactive_events", [fields])
        return int(row["id"])

    # ------------------------------------------------------------------ snapshots & digests
    def backup_to(self, dest_path: str | Path) -> Path:
        """Atomic, self-contained snapshot (``Connection.backup`` -> ``<dest>.tmp`` -> ``os.replace``).

        Safe while WAL is active; the snapshot is converted to rollback-journal mode so it is one file.
        """
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        for p in (tmp, Path(f"{tmp}-wal"), Path(f"{tmp}-shm"), Path(f"{tmp}-journal")):
            p.unlink(missing_ok=True)
        with self._lock:
            if self._depth > 0:
                raise StoreError("backup_to() inside an open transaction")
            dst = sqlite3.connect(str(tmp))
            try:
                self._conn.backup(dst)
                dst.execute("PRAGMA journal_mode=DELETE")
            finally:
                dst.close()
        os.replace(tmp, dest)
        return dest

    def content_digest(self, run_id: str) -> str:
        """sha256 over the deterministic content: generations (all columns except latency, batch id and
        timestamps), the run's cells, scores, slots, proposals and trajectory rounds (minus timestamps).

        Ledger rows, shard bookkeeping and timestamps are excluded, so an interrupted + resumed run, a
        restored snapshot + shard replay and an uninterrupted run all have the same digest.
        """
        parts: list[tuple[str, str, tuple]] = [
            (
                "generations",
                "SELECT gen_key, engine_fp, model_id, model_revision, rendered_sha, system_hash, user_hash, "
                "decoding_json, seed, nonce, response, finish_reason, n_prompt_tokens, n_completion_tokens "
                "FROM generations ORDER BY gen_key",
                (),
            ),
            (
                "cells",
                f"SELECT {', '.join(CELL_COLUMNS)} FROM cells WHERE run_id = ? "
                f"ORDER BY {', '.join(CELL_KEY_COLUMNS)}",
                (run_id,),
            ),
            (
                "scores",
                "SELECT gen_key, extractor, ext_hash, extracted, method, span_start, span_end, gold, correct "
                "FROM scores ORDER BY gen_key, extractor, gold",
                (),
            ),
            (
                "slots",
                "SELECT seed, slot, prompt_hash, created_round, parent_slot, origin FROM slots "
                "WHERE run_id = ? ORDER BY seed, slot",
                (run_id,),
            ),
            (
                "proposals",
                "SELECT seed, round, attempt, meta_prompt_hash, error_item_idxs, gen_key, parsed_prompt_hash, "
                "valid, violations FROM proposals WHERE run_id = ? ORDER BY seed, round, attempt",
                (run_id,),
            ),
            (
                "trajectory_rounds",
                "SELECT seed, round, incumbent_slot, candidate_slot, inc_dev_acc, cand_dev_acc, advanced, "
                "n_attempts, is_fallback FROM trajectory_rounds WHERE run_id = ? ORDER BY seed, round",
                (run_id,),
            ),
        ]
        h = hashlib.sha256()
        with self._lock:
            for tag, sql, params in parts:
                h.update(f"#{tag}\n".encode())
                for row in self._conn.execute(sql, params):
                    h.update(json.dumps(list(row), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
                    h.update(b"\n")
        return h.hexdigest()
