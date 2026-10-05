"""Frozen answer extractors: v1 (strict) and v2 (lenient), gold parsing and exact correctness.

The extractors are a research artifact: they define what counts as correct. Each is identified by a source
hash (``extractor_hash``) over ``common.py`` plus its version module; ``scores.ext_hash`` and the environment
fingerprints reference it, and ``tests/extractors_frozen.json`` pins it. A behavioural change therefore
requires a NEW version module (``v3.py``) registered under a new name, never an edit of v1 / v2.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from functools import cache
from importlib import resources
from pathlib import Path

from driftlab.extraction.common import (
    METHODS,
    NO_EXTRACTION,
    Extraction,
    canonical,
    find_boxes,
    gold_value,
    is_correct,
    last_hash_line,
    parse_gold,
    parse_lenient,
    select_marker,
)
from driftlab.extraction.v1 import extract as extract_v1
from driftlab.extraction.v2 import extract as extract_v2

__all__ = [
    "FROZEN_HASHES_PATH",
    "METHODS",
    "NO_EXTRACTION",
    "REGISTRY",
    "Extraction",
    "canonical",
    "extract",
    "extract_v1",
    "extract_v2",
    "extractor_hash",
    "extractor_tag",
    "extractor_tags",
    "find_boxes",
    "frozen_mismatches",
    "gold_value",
    "is_correct",
    "last_hash_line",
    "parse_gold",
    "parse_lenient",
    "select_marker",
    "verify_frozen",
]

REGISTRY: dict[str, Callable[[str], Extraction]] = {"v1": extract_v1, "v2": extract_v2}

# Pinned hashes (repo checkout only; not package data).
FROZEN_HASHES_PATH = Path(__file__).resolve().parents[3] / "tests" / "extractors_frozen.json"

_PACKAGE = "driftlab.extraction"
_SHARED_SOURCE = "common.py"


def _check(name: str) -> None:
    if name not in REGISTRY:
        raise KeyError(f"unknown extractor {name!r}; registered: {sorted(REGISTRY)}")


def extract(name: str, text: str) -> Extraction:
    """Apply the registered extractor ``name`` ("v1" | "v2") to ``text``."""
    _check(name)
    return REGISTRY[name](text)


def _hash_sources(sources: Iterable[bytes]) -> str:
    """sha256 hex of the concatenated sources after normalising line endings to ``\\n``."""
    h = hashlib.sha256()
    for src in sources:
        h.update(src.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
    return h.hexdigest()


@cache
def extractor_hash(name: str) -> str:
    """First 12 hex chars of sha256(common.py + <name>.py source bytes, line endings normalised)."""
    _check(name)
    pkg = resources.files(_PACKAGE)
    sources = (pkg.joinpath(_SHARED_SOURCE).read_bytes(), pkg.joinpath(f"{name}.py").read_bytes())
    return _hash_sources(sources)[:12]


def extractor_tag(name: str) -> str:
    """Versioned extractor id used in environment fingerprints, e.g. ``"v1@1a2b3c4d"``."""
    return f"{name}@{extractor_hash(name)[:8]}"


def extractor_tags() -> dict[str, str]:
    """``{name: extractor_tag(name)}`` for every registered extractor."""
    return {name: extractor_tag(name) for name in REGISTRY}


def frozen_mismatches(expected: Mapping[str, str]) -> dict[str, tuple[str | None, str | None]]:
    """``{name: (expected, actual)}`` for every extractor whose pinned and current hashes differ.

    Covers the union of pinned and registered names (a missing side is ``None``); empty means all frozen.
    """
    out: dict[str, tuple[str | None, str | None]] = {}
    for name in sorted(set(expected) | set(REGISTRY)):
        want = expected.get(name)
        have = extractor_hash(name) if name in REGISTRY else None
        if want != have:
            out[name] = (want, have)
    return out


def verify_frozen(path: str | Path | None = None) -> dict[str, tuple[str | None, str | None]]:
    """Compare the current hashes with the pinned JSON (default ``FROZEN_HASHES_PATH``); see above."""
    pinned = json.loads(Path(path or FROZEN_HASHES_PATH).read_text(encoding="utf-8"))
    return frozen_mismatches(pinned)
