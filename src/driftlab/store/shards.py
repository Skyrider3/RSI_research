"""Immutable, crash-safe shard log (gzip JSONL) used to restore a run from a snapshot plus newer shards.

Every committed engine chunk (and any other sharded write) produces one file
``<seq:06d>_<stage>_<sha8>.jsonl.gz`` with one line per row: ``{"table": name, "row": {...}}``. Files are
written to ``*.tmp`` and published with ``os.replace``; ``sha8`` is the prefix of the sha256 of the
(deterministic, ``mtime=0``) gzip bytes, so a truncated or edited shard is detected on replay.

Restore procedure (Colab / Drive): copy the latest DB snapshot (``Store.backup_to``) -> open it ->
:func:`replay_shards` (re-inserts every shard with ``seq > meta.last_shard_seq``) -> only then
``ShardWriter.for_store(store, dir)`` to continue writing. A new writer quarantines files with
``seq >= start_seq`` (renamed ``*.orphan``): they belong to transactions that never committed.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from driftlab.store.store import Store

SHARD_RE = re.compile(r"^(?P<seq>\d{6,})_(?P<stage>.*)_(?P<sha8>[0-9a-f]{8})\.jsonl\.gz$")
_STAGE_SAFE = re.compile(r"[^A-Za-z0-9.\-]+")


class ShardCorrupt(ValueError):
    """A shard's content does not match the checksum in its file name, or cannot be parsed."""


def _safe_stage(stage: str) -> str:
    return _STAGE_SAFE.sub("-", stage).strip("-") or "chunk"


def list_shards(dir: str | Path) -> list[tuple[int, Path]]:
    """``[(seq, path), ...]`` of published shard files, sorted by (seq, name)."""
    d = Path(dir)
    if not d.is_dir():
        return []
    out = []
    for p in d.iterdir():
        m = SHARD_RE.match(p.name)
        if m and p.is_file():
            out.append((int(m.group("seq")), p))
    return sorted(out, key=lambda t: (t[0], t[1].name))


def _load_shard(path: Path) -> tuple[dict[str, list[dict]], str]:
    m = SHARD_RE.match(path.name)
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if m is None or not sha.startswith(m.group("sha8")):
        raise ShardCorrupt(f"checksum mismatch for shard {path}")
    payload: dict[str, list[dict]] = {}
    try:
        for line in gzip.decompress(data).decode("utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                payload.setdefault(rec["table"], []).append(rec["row"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ShardCorrupt(f"cannot parse shard {path}: {e}") from e
    return payload, sha


def read_shard(path: str | Path) -> dict[str, list[dict]]:
    """Verify and parse one shard into ``{table: [row, ...]}`` (row order preserved)."""
    return _load_shard(Path(path))[0]


class ShardWriter:
    """Writes numbered immutable shard files into ``dir`` starting at ``start_seq``."""

    def __init__(self, dir: str | Path, start_seq: int = 1) -> None:
        if int(start_seq) < 1:
            raise ValueError("shard sequence numbers start at 1")
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._next = int(start_seq)
        self.orphans: list[Path] = self._quarantine_orphans()

    @classmethod
    def for_store(cls, store: Store, dir: str | Path) -> ShardWriter:
        """Writer continuing after the store's ``meta.last_shard_seq`` (replay newer shards first!)."""
        return cls(dir, int(store.get_meta("last_shard_seq") or 0) + 1)

    @property
    def next_seq(self) -> int:
        return self._next

    def _quarantine_orphans(self) -> list[Path]:
        moved = []
        for seq, p in list_shards(self.dir):
            if seq >= self._next:
                dest = p.with_name(p.name + ".orphan")
                os.replace(p, dest)
                moved.append(dest)
        for p in self.dir.glob("*.jsonl.gz.tmp"):
            p.unlink(missing_ok=True)
        if moved:
            warnings.warn(
                f"quarantined {len(moved)} uncommitted shard(s) in {self.dir} (seq >= {self._next}); "
                "if you are restoring a snapshot, call replay_shards() before creating the writer",
                stacklevel=3,
            )
        return moved

    def write(
        self, stage: str, payload: Mapping[str, Sequence[Mapping[str, Any]]]
    ) -> tuple[int, Path, int, str]:
        """Write one shard; returns ``(seq, path, n_rows, sha256)``. Rows must be JSON-serializable."""
        seq = self._next
        lines = []
        for table, rows in payload.items():
            for row in rows:
                lines.append(
                    json.dumps({"table": table, "row": dict(row)}, sort_keys=True, ensure_ascii=False)
                )
        raw = ("\n".join(lines) + "\n").encode("utf-8")
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
            gz.write(raw)
        data = buf.getvalue()
        sha = hashlib.sha256(data).hexdigest()
        path = self.dir / f"{seq:06d}_{_safe_stage(stage)}_{sha[:8]}.jsonl.gz"
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        self._next = seq + 1
        return seq, path, len(lines), sha

    def discard(self, seq: int, path: str | Path) -> None:
        """Remove a shard whose transaction failed to commit and reuse its sequence number."""
        Path(path).unlink(missing_ok=True)
        if self._next == int(seq) + 1:
            self._next = int(seq)


def replay_shards(store: Store, dir: str | Path) -> int:
    """Re-insert every shard with ``seq > meta.last_shard_seq`` (one transaction per shard, table
    policies of :class:`Store`), record it in ``shard_log`` and advance ``last_shard_seq``.

    Returns the number of rows read from the replayed shards (rows already present are ignored).
    """
    last = int(store.get_meta("last_shard_seq") or 0)
    total = 0
    prev = last
    for seq, path in list_shards(dir):
        if seq <= last:
            continue
        if seq == prev:
            warnings.warn(f"duplicate shard sequence number {seq}: {path.name}", stacklevel=2)
        elif seq != prev + 1:
            warnings.warn(f"gap in shard log before seq {seq} (last replayed {prev})", stacklevel=2)
        payload, sha = _load_shard(path)
        n_rows = sum(len(rows) for rows in payload.values())
        store.apply_shard(seq, path.name, n_rows, sha, payload)
        total += n_rows
        prev = seq
    return total
