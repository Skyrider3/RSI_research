"""GSM8K items: the committed first-200 snapshot (verified) or the pinned Hugging Face revision.

* ``dev`` = first ``data.dev.n`` rows of the TRAIN split (Phase A only); ``eval`` = first ``data.eval.n`` rows
  of the TEST split (Phase B only). Rows are taken in the published order (no shuffling).
* The snapshot (``data/gsm8k/{split}_first200.jsonl``) is verified against ``snapshot.json``: dataset id,
  config and revision must equal the config's pins and every file's sha256 must match; any mismatch raises
  :class:`DataIntegrityError` (never a silent fallback).
* If the snapshot is missing, or more rows are requested than it holds, the split's parquet file is
  downloaded from the Hub at the pinned revision (``huggingface_hub``, optional ``[data]`` extra) and its
  sha256 is checked against ``snapshot.json["source_files"]`` when known.
* :class:`DevSplit` and :class:`EvalSplit` are distinct types so Phase A can refuse the eval split.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, ClassVar, Literal

from driftlab.config import ExperimentConfig

DEV_SPLIT = "train"
EVAL_SPLIT = "test"
SNAPSHOT_MANIFEST = "snapshot.json"
UNICODE_MINUS = "−"


class DataIntegrityError(RuntimeError):
    """The dataset on disk / from the Hub does not match the pinned revision or recorded hashes."""


class LeakageError(ValueError):
    """An item of the wrong split reached a stage that must never see it (e.g. eval items in Phase A)."""


# --------------------------------------------------------------------------- gold parsing


def canonical_number(s: str) -> str:
    """Canonical numeric string: commas/spaces removed, integral values as ints, decimals without trailing
    zeros (``"1,234"`` -> ``"1234"``, ``"5.00"`` -> ``"5"``, ``"0.50"`` -> ``"0.5"``). Raises ``ValueError``."""
    raw = str(s).strip().replace(",", "").replace(" ", "").replace(UNICODE_MINUS, "-")
    try:
        d = Decimal(raw)
    except InvalidOperation:
        raise ValueError(f"not a number: {s!r}") from None
    if not d.is_finite():
        raise ValueError(f"not a finite number: {s!r}")
    if d == d.to_integral_value():
        return str(int(d))
    return format(d.normalize(), "f")


def parse_gold(answer_text: str) -> str:
    """Canonical gold of a GSM8K reference solution: the number after the last ``####``."""
    if "####" not in answer_text:
        raise ValueError("GSM8K answer has no '####' marker")
    return canonical_number(answer_text.split("####")[-1])


# --------------------------------------------------------------------------- items and splits


@dataclass(frozen=True)
class Item:
    split: str  # "train" | "test"
    idx: int  # row index in the published split
    question: str
    answer_text: str  # full reference solution
    gold: str  # canonical numeric string, e.g. "72"

    def to_row(self) -> dict[str, Any]:
        """Row for ``Store.put_items`` / the ``items`` table."""
        return {
            "split": self.split,
            "idx": self.idx,
            "question": self.question,
            "answer_text": self.answer_text,
            "gold": self.gold,
        }


@dataclass(frozen=True)
class _Split:
    items: tuple[Item, ...]
    _by_idx: dict[int, Item] = field(init=False, repr=False, compare=False, hash=False)

    expected_split: ClassVar[str] = ""

    def __post_init__(self) -> None:
        items = tuple(self.items)
        object.__setattr__(self, "items", items)
        bad = [it for it in items if it.split != self.expected_split]
        if bad:
            raise LeakageError(
                f"{type(self).__name__} accepts only split={self.expected_split!r} items; got "
                f"{len(bad)} item(s) of split {sorted({it.split for it in bad})}"
            )
        index = {it.idx: it for it in items}
        if len(index) != len(items):
            raise ValueError(f"{type(self).__name__} has duplicate item indices")
        object.__setattr__(self, "_by_idx", index)

    @property
    def split(self) -> str:
        return self.expected_split

    def questions(self) -> list[str]:
        return [it.question for it in self.items]

    def golds(self) -> list[str]:
        return [it.gold for it in self.items]

    def idxs(self) -> list[int]:
        return [it.idx for it in self.items]

    def by_idx(self, idx: int) -> Item:
        """The item with dataset row index ``idx`` (``KeyError`` if absent)."""
        return self._by_idx[idx]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Item]:
        return iter(self.items)

    def __getitem__(self, pos: int) -> Item:
        return self.items[pos]


@dataclass(frozen=True)
class DevSplit(_Split):
    """Development items (train split). The only split Phase A may see."""

    expected_split: ClassVar[str] = DEV_SPLIT


@dataclass(frozen=True)
class EvalSplit(_Split):
    """Evaluation items (test split). Phase B only; never passed to the trajectory or the proposer."""

    expected_split: ClassVar[str] = EVAL_SPLIT


# --------------------------------------------------------------------------- snapshot verification


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_dir(cfg: ExperimentConfig) -> Path:
    return cfg.resolve_path(cfg.data.snapshot_dir)


def snapshot_filename(split: str) -> str:
    return f"{split}_first200.jsonl"


def source_filename(cfg: ExperimentConfig, split: str) -> str:
    """Path of the split's parquet file inside the Hub dataset repo."""
    return f"{cfg.data.config}/{split}-00000-of-00001.parquet"


def _read_manifest(directory: Path) -> dict | None:
    path = directory / SNAPSHOT_MANIFEST
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise DataIntegrityError(f"{path} is not valid JSON: {e}") from None


def _check_manifest_pins(cfg: ExperimentConfig, manifest: Mapping[str, Any], where: Path) -> None:
    for key, want in (
        ("dataset", cfg.data.dataset),
        ("config", cfg.data.config),
        ("revision", cfg.data.revision),
    ):
        have = manifest.get(key)
        if have != want:
            raise DataIntegrityError(
                f"{where / SNAPSHOT_MANIFEST}: {key} is {have!r} but the config pins {want!r}"
            )


def _split_section(cfg: ExperimentConfig, which: str) -> tuple[str, int]:
    if which == "dev":
        sec = cfg.data.dev
    elif which == "eval":
        sec = cfg.data.eval
    else:
        raise ValueError(f"which must be 'dev' or 'eval', got {which!r}")
    if sec.n < 1:
        raise ValueError(f"data.{which}.n must be >= 1, got {sec.n}")
    return sec.split, sec.n


def _item_from_row(row: Mapping[str, Any], split: str, pos: int, origin: str) -> Item:
    if "split" in row and row["split"] != split:
        raise DataIntegrityError(f"{origin}: row {pos} has split {row['split']!r}, expected {split!r}")
    idx = int(row.get("idx", pos))
    if idx != pos:
        raise DataIntegrityError(f"{origin}: row {pos} has idx {idx} (rows must be in published order)")
    question, answer = str(row["question"]), str(row["answer"])
    try:
        gold = parse_gold(answer)
    except ValueError as e:
        raise DataIntegrityError(f"{origin}: row {pos}: {e}") from None
    if "gold" in row:
        try:
            stored = canonical_number(str(row["gold"]))
        except ValueError:
            stored = None
        if stored != gold:
            raise DataIntegrityError(
                f"{origin}: row {pos} stores gold {row['gold']!r} but its answer parses to {gold!r}"
            )
    return Item(split=split, idx=idx, question=question, answer_text=answer, gold=gold)


def _load_snapshot_rows(cfg: ExperimentConfig, split: str, n: int) -> list[Item] | None:
    """First ``n`` verified snapshot items, or ``None`` if the snapshot is absent or too small."""
    directory = snapshot_dir(cfg)
    manifest = _read_manifest(directory)
    name = snapshot_filename(split)
    path = directory / name
    if manifest is None or not path.is_file():
        return None
    _check_manifest_pins(cfg, manifest, directory)
    entry = (manifest.get("files") or {}).get(name)
    if not entry or "sha256" not in entry:
        raise DataIntegrityError(f"{directory / SNAPSHOT_MANIFEST} has no sha256 for {name}")
    if entry.get("split", split) != split:
        raise DataIntegrityError(f"{name} is recorded as split {entry.get('split')!r}, expected {split!r}")
    actual = _sha256_file(path)
    if actual != entry["sha256"]:
        raise DataIntegrityError(
            f"{path}: sha256 {actual} does not match the recorded {entry['sha256']} (file modified?)"
        )
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if "n" in entry and int(entry["n"]) != len(lines):
        raise DataIntegrityError(f"{path}: {len(lines)} rows but snapshot.json records {entry['n']}")
    if n > len(lines):
        return None
    return [_item_from_row(json.loads(ln), split, i, str(path)) for i, ln in enumerate(lines[:n])]


def _hf_hub_download(repo_id: str, filename: str, revision: str) -> str:
    """Download one file of a Hub dataset repo (lazy optional import; patched in tests)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "the GSM8K snapshot is missing or too small and huggingface_hub is not installed; "
            "install the data extra (pip install -e '.[data]') or restore data/gsm8k/"
        ) from None
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision)


def _load_hub_rows(cfg: ExperimentConfig, split: str, n: int) -> list[Item]:
    filename = source_filename(cfg, split)
    local = Path(_hf_hub_download(cfg.data.dataset, filename, cfg.data.revision))
    manifest = _read_manifest(snapshot_dir(cfg))
    if manifest is not None:
        _check_manifest_pins(cfg, manifest, snapshot_dir(cfg))
        want = ((manifest.get("source_files") or {}).get(filename) or {}).get("sha256")
        if want is not None:
            actual = _sha256_file(local)
            if actual != want:
                raise DataIntegrityError(
                    f"{filename}@{cfg.data.revision}: sha256 {actual} != recorded {want}"
                )
    try:
        import pandas as pd

        df = pd.read_parquet(local, columns=["question", "answer"])
    except ImportError:  # pragma: no cover - depends on the environment
        raise RuntimeError("reading the GSM8K parquet needs pyarrow (pip install -e '.[data]')") from None
    if n > len(df):
        raise ValueError(f"requested {n} {split} items but {filename} has only {len(df)} rows")
    rows = df.head(n).to_dict("records")
    return [_item_from_row(r, split, i, f"{filename}@{cfg.data.revision}") for i, r in enumerate(rows)]


# --------------------------------------------------------------------------- public loaders


def load_items(cfg: ExperimentConfig, which: Literal["dev", "eval"]) -> list[Item]:
    """First ``n`` items of the dev (train) or eval (test) split, verified against the pinned snapshot."""
    split, n = _split_section(cfg, which)
    items = _load_snapshot_rows(cfg, split, n)
    if items is None:
        items = _load_hub_rows(cfg, split, n)
    return items


def load_dev(cfg: ExperimentConfig) -> DevSplit:
    return DevSplit(tuple(load_items(cfg, "dev")))


def load_eval(cfg: ExperimentConfig) -> EvalSplit:
    return EvalSplit(tuple(load_items(cfg, "eval")))


def load_split(cfg: ExperimentConfig, which: Literal["dev", "eval"]) -> DevSplit | EvalSplit:
    """``load_dev`` / ``load_eval`` by name (ARCHITECTURE module-map spelling)."""
    if which == "dev":
        return load_dev(cfg)
    if which == "eval":
        return load_eval(cfg)
    raise ValueError(f"which must be 'dev' or 'eval', got {which!r}")


def user_message(cfg: ExperimentConfig, question: str) -> str:
    """The answer request's user turn for a question (``trajectory.user_template``)."""
    return cfg.trajectory.user_template.format(question=question)


def answer_key(cfg: ExperimentConfig) -> dict[str, str]:
    """``{user message: gold}`` over BOTH splits (for the mock backend only; never shown to the proposer)."""
    key: dict[str, str] = {}
    for which in ("dev", "eval"):
        for it in load_items(cfg, which):  # type: ignore[arg-type]
            msg = user_message(cfg, it.question)
            if key.get(msg, it.gold) != it.gold:
                raise DataIntegrityError(f"question appears with two different golds: {it.question[:60]!r}")
            key[msg] = it.gold
    return key


def snapshot_info(cfg: ExperimentConfig) -> dict[str, Any]:
    """Dataset pins and snapshot file hashes for provenance (does not raise on a bad snapshot)."""
    directory = snapshot_dir(cfg)
    manifest = _read_manifest(directory) or {}
    recorded = manifest.get("files") or {}
    files: dict[str, dict[str, Any]] = {}
    for split in (cfg.data.dev.split, cfg.data.eval.split):
        name = snapshot_filename(split)
        path = directory / name
        actual = _sha256_file(path) if path.is_file() else None
        want = (recorded.get(name) or {}).get("sha256")
        files[name] = {"sha256": actual, "recorded_sha256": want, "ok": actual is not None and actual == want}
    pins_ok = all(
        manifest.get(k) == v
        for k, v in (
            ("dataset", cfg.data.dataset),
            ("config", cfg.data.config),
            ("revision", cfg.data.revision),
        )
    )
    return {
        "dataset": cfg.data.dataset,
        "config": cfg.data.config,
        "revision": cfg.data.revision,
        "snapshot_dir": cfg.data.snapshot_dir,
        "snapshot_present": bool(manifest),
        "snapshot_revision": manifest.get("revision"),
        "pins_match": pins_ok,
        "files": files,
        "source_files": {k: (v or {}).get("sha256") for k, v in (manifest.get("source_files") or {}).items()},
        "dev": {"split": cfg.data.dev.split, "n": cfg.data.dev.n},
        "eval": {"split": cfg.data.eval.split, "n": cfg.data.eval.n},
    }


__all__ = [
    "DEV_SPLIT",
    "EVAL_SPLIT",
    "DataIntegrityError",
    "DevSplit",
    "EvalSplit",
    "Item",
    "LeakageError",
    "answer_key",
    "canonical_number",
    "load_dev",
    "load_eval",
    "load_items",
    "load_split",
    "parse_gold",
    "snapshot_info",
    "user_message",
]
