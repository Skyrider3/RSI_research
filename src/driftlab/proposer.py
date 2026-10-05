"""Proposer logic for Phase A: meta-prompt construction, candidate parsing / validation and the fallback edit.

Everything here is a pure function of its inputs (plus the packaged templates), so the meta-prompt text and
therefore the proposer's cache key are reproducible. Only DEV (train-split) errors can reach the proposer:
:class:`DevError` refuses any other split and :func:`build_meta_prompt` refuses anything that is not a
``DevError``.
"""

from __future__ import annotations

import random
import re
import string
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import cache
from importlib import resources

import yaml

from driftlab import keys
from driftlab.backends.base import GenRequest
from driftlab.config import ProposerSection
from driftlab.data import DEV_SPLIT, LeakageError
from driftlab.environments import Decoding
from driftlab.prompting import PROPOSER_SYSTEM, fill, load_prompt_file

ERROR_BLOCK_TEMPLATE = "error_block_v1.txt"
FALLBACK_EDITS_FILE = "fallback_edits.yaml"
ELLIPSIS = " … "
BLOCK_SEPARATOR = "\n\n"

# Validation violations, in reporting order.
VIOLATIONS: tuple[str, ...] = (
    "missing",
    "too_short",
    "too_long",
    "no_format_instruction",
    "duplicate",
    "leaks_gold",
    "contains_tags",
)
LEAK_MIN_DIGITS = 3

_PROMPT_PAIR_RE = re.compile(r"<\s*prompt\s*>(.*?)<\s*/\s*prompt\s*>", re.S | re.I)
_OPEN_PROMPT_RE = re.compile(r"<\s*prompt\s*>", re.I)
_LEFTOVER_TAG_RE = re.compile(r"<\s*/?\s*(?:current_)?prompt\s*>", re.I)
_NUMBER_TOKEN_RE = re.compile(r"(?<![\d.,])\d+(?:[.,]\d+)*")
_GROUPED_RE = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
_PLAIN_RE = re.compile(r"^\d+(?:\.\d+)?$")
_EDGE_PUNCT = string.punctuation + "…“”‘’«»"


@dataclass(frozen=True)
class DevError:
    """A dev item the incumbent answered incorrectly (shown to the proposer)."""

    idx: int
    question: str
    response: str
    gold: str
    split: str = DEV_SPLIT

    def __post_init__(self) -> None:
        if self.split != DEV_SPLIT:
            raise LeakageError(
                f"DevError must come from the {DEV_SPLIT!r} split; got split={self.split!r} (item {self.idx}). "
                "The eval split must never reach the proposer."
            )


def _check_dev_errors(errors: Sequence[DevError]) -> None:
    for e in errors:
        if not isinstance(e, DevError):
            raise TypeError(f"proposer errors must be DevError instances, got {type(e).__name__}")
        if e.split != DEV_SPLIT:  # defends against object.__setattr__ tampering
            raise LeakageError(f"non-dev item {e.idx} (split={e.split!r}) passed to the proposer")


def sample_errors(errors: Sequence[DevError], k: int, seed: int, round_: int) -> list[DevError]:
    """``k`` errors sampled with ``random.Random(rng_seed(seed, "errors", round_))`` from the errors sorted by
    idx (sample order kept); all of them, sorted by idx, if there are fewer than ``k``."""
    _check_dev_errors(errors)
    ordered = sorted(errors, key=lambda e: e.idx)
    if k <= 0:
        return []
    if len(ordered) < k:
        return ordered
    return random.Random(keys.rng_seed(seed, "errors", round_)).sample(ordered, k)


def truncate_response(text: str, head: int, tail: int) -> str:
    """``text`` if it is short, else its first ``head`` chars + ``" … "`` + its last ``tail`` chars."""
    if head < 0 or tail < 0:
        raise ValueError("head and tail must be >= 0")
    if len(text) <= head + tail + len(ELLIPSIS):
        return text
    return text[:head] + ELLIPSIS + (text[-tail:] if tail else "")


def shown_golds(errors: Sequence[DevError]) -> list[str]:
    """Gold answers displayed in a meta-prompt (input to the ``leaks_gold`` check)."""
    return [e.gold for e in errors]


def build_meta_prompt(current_prompt: str, errors: Sequence[DevError], pcfg: ProposerSection) -> str:
    """Fill ``pcfg.template`` with the incumbent and one numbered ``error_block_v1.txt`` block per error."""
    _check_dev_errors(errors)
    block = load_prompt_file(ERROR_BLOCK_TEMPLATE)
    blocks = BLOCK_SEPARATOR.join(
        fill(
            block,
            i=i,
            question=e.question,
            response=truncate_response(e.response, pcfg.response_head_chars, pcfg.response_tail_chars),
            gold=e.gold,
        )
        for i, e in enumerate(errors, start=1)
    )
    return fill(
        load_prompt_file(pcfg.template),
        current_prompt=current_prompt,
        n_errors=len(errors),
        error_blocks=blocks,
    )


def proposer_request(meta_prompt: str, decoding: Decoding, seed: int) -> GenRequest:
    """The proposer call (``seed`` = ``keys.proposer_seed(run_seed, round_, attempt)``)."""
    return GenRequest(system=PROPOSER_SYSTEM, user=meta_prompt, decoding=decoding, seed=seed)


def parse_candidate(text: str) -> str | None:
    """Content of the LAST ``<prompt>…</prompt>`` pair (tags case-insensitive), stripped.

    ``None`` if there is no complete pair, the content is empty, or it contains another ``<prompt>`` tag
    (nested tags, e.g. ``<prompt><prompt>x</prompt></prompt>``).
    """
    matches = list(_PROMPT_PAIR_RE.finditer(text or ""))
    if not matches:
        return None
    content = matches[-1].group(1)
    if _OPEN_PROMPT_RE.search(content):
        return None
    content = content.strip()
    return content or None


def normalize_prompt(text: str) -> str:
    """Lower-cased, whitespace-collapsed text without leading/trailing punctuation (duplicate detection)."""
    return re.sub(r"\s+", " ", text.lower()).strip().strip(_EDGE_PUNCT + " ")


def _token_values(token: str) -> list[Decimal]:
    """Numeric value(s) of a digit token: one value for ``1234``/``1,234``/``12.5``; parts otherwise."""
    if _GROUPED_RE.match(token) or _PLAIN_RE.match(token):
        return [Decimal(token.replace(",", ""))]
    out = []
    for part in token.split(","):
        pieces = [part] if _PLAIN_RE.match(part) else part.split(".")
        out.extend(Decimal(p) for p in pieces if p)
    return out


def _gold_value(gold: str) -> Decimal | None:
    g = str(gold).strip().replace(",", "").lstrip("+-−")
    if sum(c.isdigit() for c in g) < LEAK_MIN_DIGITS:
        return None
    try:
        return Decimal(g)
    except InvalidOperation:
        return None


def leaked_golds(candidate: str, golds: Sequence[str]) -> list[str]:
    """Shown golds with >= 3 digits that appear in ``candidate`` as a standalone number (value match, so
    ``1,234`` and ``1234.0`` both match gold ``1234``; ``12345`` or ``1.234`` do not)."""
    values = {v for tok in _NUMBER_TOKEN_RE.findall(candidate) for v in _token_values(tok)}
    out = []
    for g in golds:
        gv = _gold_value(g)
        if gv is not None and gv in values and g not in out:
            out.append(g)
    return out


def validate_candidate(
    candidate: str | None,
    prior_prompts: Sequence[str],
    shown_golds: Sequence[str],
    pcfg: ProposerSection,
) -> list[str]:
    """Violations (subset of :data:`VIOLATIONS`, in that order); empty means valid.

    ``prior_prompts``: every earlier slot text of this seed (incl. the incumbent); ``shown_golds``: the gold
    answers displayed in the meta-prompt.
    """
    if candidate is None:
        return ["missing"]
    out: list[str] = []
    if len(candidate) < pcfg.min_chars:
        out.append("too_short")
    if len(candidate) > pcfg.max_chars:
        out.append("too_long")
    if pcfg.require_format_instruction and "\\boxed" not in candidate:
        out.append("no_format_instruction")
    norm = normalize_prompt(candidate)
    if any(norm == normalize_prompt(p) for p in prior_prompts):
        out.append("duplicate")
    if leaked_golds(candidate, shown_golds):
        out.append("leaks_gold")
    if _LEFTOVER_TAG_RE.search(candidate):
        out.append("contains_tags")
    return out


@cache
def load_fallback_edits(name: str = FALLBACK_EDITS_FILE) -> tuple[str, ...]:
    """The packaged fallback sentences, in file order."""
    text = resources.files("driftlab.prompts").joinpath(name).read_text(encoding="utf-8")
    edits = (yaml.safe_load(text) or {}).get("edits") or []
    return tuple(str(e).strip() for e in edits if str(e).strip())


def fallback_candidate(incumbent: str, seed: int, round_: int, prior_prompts: Sequence[str]) -> str:
    """Incumbent + newline + one fallback sentence not already present, chosen with
    ``random.Random(rng_seed(seed, "fallback", round_))`` among those whose result is not a duplicate of a
    prior prompt. If none qualifies, appends ``"(Revision {round_}.)"`` instead."""
    base = incumbent.rstrip()
    norm_inc = normalize_prompt(incumbent)
    prior = {normalize_prompt(p) for p in prior_prompts}
    options = []
    for sentence in load_fallback_edits():
        if normalize_prompt(sentence) in norm_inc:
            continue
        result = f"{base}\n{sentence}"
        if normalize_prompt(result) not in prior:
            options.append(result)
    if options:
        return random.Random(keys.rng_seed(seed, "fallback", round_)).choice(options)
    result = f"{base}\n(Revision {round_}.)"
    variant = 2
    while normalize_prompt(result) in prior:  # practically unreachable; keeps the result unique
        result = f"{base}\n(Revision {round_}, variant {variant}.)"
        variant += 1
    return result


__all__ = [
    "VIOLATIONS",
    "DevError",
    "LeakageError",
    "build_meta_prompt",
    "fallback_candidate",
    "leaked_golds",
    "load_fallback_edits",
    "normalize_prompt",
    "parse_candidate",
    "proposer_request",
    "sample_errors",
    "shown_golds",
    "truncate_response",
    "validate_candidate",
]
