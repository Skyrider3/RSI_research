"""Experiment store: SQLite (WAL) database plus an immutable shard log for crash-safe restores."""

from __future__ import annotations

from driftlab.store.shards import ShardCorrupt, ShardWriter, list_shards, read_shard, replay_shards
from driftlab.store.store import (
    ConfigMismatch,
    EngineMismatch,
    SlotConflict,
    Store,
    StoreError,
    to_json,
    utc_now,
)

__all__ = [
    "ConfigMismatch",
    "EngineMismatch",
    "ShardCorrupt",
    "ShardWriter",
    "SlotConflict",
    "Store",
    "StoreError",
    "list_shards",
    "read_shard",
    "replay_shards",
    "to_json",
    "utc_now",
]
