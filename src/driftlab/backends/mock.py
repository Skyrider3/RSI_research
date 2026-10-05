"""Deterministic, structured synthetic stand-in for Qwen2.5-1.5B-Instruct (SYNTHETIC: never report its numbers).

The mock exists so that every phenomenon the study measures actually occurs in CI, tests and the labelled
synthetic dashboard demo: prompt quality differences, sampling noise, emulated GPU nondeterminism on physical
greedy reruns, answer formats that only the lenient extractor parses, truncation, and proposer outputs that
are sometimes malformed.

Answer mode (``system`` = prompt under evaluation, ``user`` = question)::

    solved  iff  q(p) - d_n + 0.5 * z(p, n) + eps > 0
    q(p)  = base_quality + sum(regex feature weights of p) + jitter(p),  jitter in [-0.15, 0.15]
    d_n   ~ N(DIFFICULTY_SHIFT, 1) from a hash of the question;  z ~ N(0, 1) from a hash of (p, n)
    eps   = 0 (greedy)  |  N(0, sample_sd * T / 0.2) seeded by the request seed (sampling)

Physical greedy reruns (``nonce`` set) flip ``solved`` with probability ``greedy_flip_rate`` (and, with the
same probability, change only the wording), seeded by ``(nonce, prompt, question)``; with a flip rate of 0
they are byte-identical to the creation generation. The final-answer format is drawn from a categorical
distribution that depends on the prompt's format instructions (and, under sampling, on the seed).

Proposer mode (``system == driftlab.prompting.PROPOSER_SYSTEM``): 1-2 seeded snippet edits of the incumbent
prompt, returned between ``<prompt>`` tags; with probability ``malformed_proposal_rate`` the proposal is
malformed (missing / empty / nested tags, no ``\\boxed`` instruction, or a copied gold answer).

Every output is a pure function of (system, user, decoding parameters, seed if sampling, nonce) and the
config: per-request hash-derived streams, no global RNG state, independent of batch composition and order.
Randomness uses only sha256 (``driftlab.keys``) and ``random.Random.random()``, whose sequence Python
guarantees across versions, so cached generations stay valid across numpy/Python upgrades.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist

from driftlab.backends.base import Backend, GenRequest, GenResult
from driftlab.config import MockSection
from driftlab.environments import Decoding
from driftlab.keys import sha256_json
from driftlab.prompting import PROPOSER_SYSTEM, extract_current_prompt

MOCK_VERSION = "1"  # bump whenever the generative model below changes behaviour (it is in engine_info)

# --------------------------------------------------------------------------- calibration constants

QUALITY_WEIGHTS: dict[str, float] = {
    "step": 0.25,  # step-by-step / show your work
    "verify": 0.15,  # double-check / verify
    "read": 0.10,  # read the question carefully
    "units_track": 0.05,  # keep track of / convert units
    "brief": -0.30,  # answer briefly
    "intuition": -0.30,  # use intuition instead of calculation
    "long": -0.10,  # prompt longer than LONG_PROMPT_WORDS words
}
PROMPT_JITTER = 0.15  # per-prompt hash jitter, uniform in [-PROMPT_JITTER, PROMPT_JITTER]
INTERACTION_SCALE = 0.5  # weight of the prompt x item interaction z ~ N(0, 1)
DIFFICULTY_SHIFT = 0.15  # mean of d_n; puts the initial prompt at ~70% (lenient) accuracy
LONG_PROMPT_WORDS = 180
TOKENS_PER_WORD = 1.3
REFERENCE_TEMPERATURE = 0.2  # sampling noise sd = sample_sd * T / REFERENCE_TEMPERATURE

# Final-answer formats. Strict-parser compatible: "box" and "hash".
FORMATS: tuple[str, ...] = ("box", "comma", "currency", "units", "answer_is", "bold", "hash", "truncated")
STRICT_FORMATS: frozenset[str] = frozenset({"box", "hash"})
_FAMILY_BOXED_PROMPT = (("boxed", 0.93), ("hash", 0.03), ("answer_is", 0.02), ("bold", 0.02))
_FAMILY_PLAIN_PROMPT = (("boxed", 0.955), ("hash", 0.025), ("answer_is", 0.01), ("bold", 0.01))
_FAMILY_NO_BOX_PROMPT = (("boxed", 0.20), ("hash", 0.10), ("answer_is", 0.45), ("bold", 0.25))
_FORMAT_HEAT_NOISE = 0.25  # under sampling, non-strict probabilities are scaled by 1 + 0.25 * T / 0.2
FORMAT_ITEM_CORRELATION = 0.6  # share of the format draw's latent variance that is question-level
_TRUNC_BASE, _TRUNC_UNSOLVED, _TRUNC_STEP, _TRUNC_HEAT = 0.005, 0.02, 0.004, 0.004


@dataclass(frozen=True)
class Snippet:
    """A sentence the mock proposer can add to / substitute into a prompt, with its known effect."""

    text: str
    weight: float  # relative selection weight
    effect: str  # "+quality" | "-quality" | "+format" (raises strict compliance) | "-format" | "neutral"


SNIPPETS: tuple[Snippet, ...] = (
    Snippet("Think step by step and show your work.", 1.0, "+quality"),
    Snippet("Double-check each arithmetic step before answering.", 1.0, "+quality"),
    Snippet("Read the question carefully and identify what is asked.", 1.0, "+quality"),
    Snippet("Keep track of units and convert them when needed.", 0.7, "+quality"),
    Snippet("After solving, verify the answer against the conditions in the problem.", 0.7, "+quality"),
    Snippet(
        "Give the final answer as a plain number with no commas, units, or currency symbols inside \\boxed{}.",
        0.8,
        "+format",
    ),
    Snippet("Answer as briefly as possible.", 0.6, "-quality"),
    Snippet("Use your intuition rather than lengthy calculations.", 0.5, "-quality"),
    Snippet("State units in the final answer.", 0.6, "-format"),
    Snippet("Write each step on its own line.", 0.6, "neutral"),
    Snippet("Be clear and organized in your explanation.", 0.6, "neutral"),
)
MALFORMED_KINDS: tuple[str, ...] = ("missing_tags", "empty_tags", "no_boxed", "nested", "copies_gold")
_EDIT_OPS = (("add", 0.55), ("replace", 0.25), ("remove", 0.20))
_EDIT_OPS_LONG = (("add", 0.2), ("replace", 0.4), ("remove", 0.4))  # incumbent over 150 words

_DEFAULT_PROMPT = (
    "You are a helpful assistant that solves grade-school math word problems. "
    "Solve the problem step by step, and put your final answer within \\boxed{}."
)
_NO_BOX_FALLBACK = "You are a careful assistant that solves grade-school math word problems."
_PREAMBLES = (
    "Here is an improved prompt:",
    "Improved system prompt:",
    "The errors suggest the assistant needs clearer guidance. Here is an improved prompt:",
)
_OPENERS = (
    "Let's work through the problem.",
    "First, identify the given quantities.",
    "We start from the numbers in the problem.",
    "Let me organize the information.",
)
_SUMMARIES = ("So the result is {v}.", "Putting it together gives {v}.", "This gives a total of {v}.")
_CALC_TEMPLATES = {
    "+": ("{a} + {b} = {r}.", "Adding these, {a} + {b} = {r}.", "Together that is {a} + {b} = {r}."),
    "-": ("{a} - {b} = {r}.", "Subtracting, {a} - {b} = {r}.", "That leaves {a} - {b} = {r}."),
    "*": ("{a} * {b} = {r}.", "Multiplying, {a} * {b} = {r}.", "So {a} * {b} = {r}."),
    "/": ("{a} / {b} = {r}.", "Dividing, {a} / {b} = {r}.", "Each share is {a} / {b} = {r}."),
}
_LOOP_LINES = ("Wait, let me recompute that.", "Let me go over the previous step again.", "Hmm, recounting.")
_BOX_LINES = ("The final answer is {box}.", "Therefore, the final answer is {box}.", "{box}")

# --------------------------------------------------------------------------- prompt features

_I = re.IGNORECASE
_BOXED_RE = re.compile(r"\\boxed")
_FEATURE_RES: dict[str, re.Pattern[str]] = {
    "step": re.compile(
        r"step[- ]by[- ]step|show (?:all |each |every )?(?:of )?(?:your |the )?(?:work|steps?)\b|work through",
        _I,
    ),
    "verify": re.compile(r"double[- ]check|\bverify|\bre-?check|\bcheck (?:each|every|your|the)\b", _I),
    "read": re.compile(
        r"read (?:the |each )?(?:question|problem)s?\s+carefully|carefully read|identify (?:exactly )?what",
        _I,
    ),
    "units_track": re.compile(
        r"(?:track|convert|label|intermediate|each)[^.\n]{0,40}\bunits?\b"
        r"|\bunits?\b[^.\n]{0,40}\b(?:track|convert|consistent)",
        _I,
    ),
    "brief": re.compile(r"\bbrief(?:ly)?\b|\bas short as possible\b|\bconcise(?:ly)?\b", _I),
    "intuition": re.compile(
        r"\bintuition|\bgut feeling|\bwithout (?:writing|showing) (?:out )?(?:the |your )?", _I
    ),
    "units_format": re.compile(
        r"\b(?:state|include|report|mention)\b[^.\n]{0,25}\bunits?\b|\bunits? in (?:the|your) final answer",
        _I,
    ),
}
_PLAIN_RE = re.compile(
    r"plain number|no commas|without (?:any )?(?:commas|units|(?:the )?currency)|no (?:units|currency)"
    r"|only the number",
    _I,
)
_NEGATION_RE = re.compile(r"\b(?:not|never|avoid|don't|no)\b", _I)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class PromptFeatures:
    """Regex-detected properties of a system prompt (what the mock 'understands')."""

    step: bool
    verify: bool
    read: bool
    units_track: bool
    brief: bool
    intuition: bool
    long: bool
    boxed: bool  # mentions \boxed (format instruction present)
    plain: bool  # asks for a plain number / no commas / no units (raises strict compliance)
    units_format: bool  # asks to state units in the answer (lowers strict compliance)
    n_words: int


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _affirmed(rx: re.Pattern[str], text: str) -> bool:
    """True if ``rx`` matches in some clause that does not negate it ("do not answer briefly")."""
    for sent in _sentences(text):
        for m in rx.finditer(sent):
            clause = re.split(r"[;,:]", sent[: m.start()])[-1]
            if not _NEGATION_RE.search(clause):
                return True
    return False


@lru_cache(maxsize=8192)
def prompt_features(system: str) -> PromptFeatures:
    """Detect the quality and format features of a system prompt."""
    n_words = len(system.split())
    found = {name: _affirmed(rx, system) for name, rx in _FEATURE_RES.items()}
    return PromptFeatures(
        long=n_words > LONG_PROMPT_WORDS,
        boxed=bool(_BOXED_RE.search(system)),
        plain=bool(_PLAIN_RE.search(system)),
        n_words=n_words,
        **found,
    )


def prompt_quality(system: str, base_quality: float = MockSection().base_quality) -> float:
    """``q(p)`` = base quality + feature weights + per-prompt hash jitter in [-0.15, 0.15]."""
    f = prompt_features(system)
    q = base_quality + sum(w for name, w in QUALITY_WEIGHTS.items() if getattr(f, name))
    return q + PROMPT_JITTER * (2.0 * _uniform("jitter", system) - 1.0)


# --------------------------------------------------------------------------- hashing helpers

_NORMAL = NormalDist()


def _hex(*parts: object) -> str:
    return sha256_json(list(parts))


def _uniform(*parts: object) -> float:
    """Uniform in (0, 1) derived from sha256 of ``parts`` (52 bits)."""
    return (int(_hex(*parts)[:13], 16) + 0.5) / 2.0**52


def _normal(*parts: object) -> float:
    return _NORMAL.inv_cdf(_uniform(*parts))


class _Stream:
    """Deterministic draw sequence keyed by sha256 of ``parts`` (uses only ``random.Random.random``)."""

    def __init__(self, *parts: object) -> None:
        self._rng = random.Random(int(_hex(*parts), 16))

    def uniform(self) -> float:
        return self._rng.random()

    def below(self, n: int) -> int:
        return min(int(self._rng.random() * n), n - 1)

    def choice(self, seq: Sequence):
        return seq[self.below(len(seq))]

    def weighted(self, pairs: Sequence[tuple[str, float]]) -> str:
        return _pick(pairs, self._rng.random())


def _decode_key(dec: Decoding, seed: int | None) -> tuple:
    """Fields of a request that determine its draw (the seed only when sampling)."""
    if dec.is_greedy:
        return ("greedy", dec.repetition_penalty)
    return ("sample", dec.temperature, dec.top_p, dec.top_k, dec.repetition_penalty, seed)


# --------------------------------------------------------------------------- numbers and items

_NUMBER_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
_HOW_MANY_RE = re.compile(r"\bhow many\s+(?:more\s+|fewer\s+|total\s+|of\s+the\s+|of\s+)?([a-z]+)", _I)
_UNIT_STOP = frozenset(
    {"more", "fewer", "of", "the", "in", "are", "is", "was", "were", "does", "do", "did", "will", "can"}
    | {"could", "would", "should", "has", "have", "had", "total", "times", "a", "an"}
)
_GOLD_BLOCK_RE = re.compile(r"### Correct final answer\s*\n+\s*([^\n]+)")


def _parse_number(s: object) -> Fraction | None:
    """Exact value of a decimal string ("1,234", "5.5"), a "p/q" string or a number; ``None`` otherwise."""
    if s is None:
        return None
    text = str(s).replace(",", "").strip()
    try:
        return Fraction(Decimal(text))
    except (InvalidOperation, ValueError, OverflowError):  # NaN / Infinity / not a decimal
        pass
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def _fmt(v: Fraction, commas: bool = False) -> str:
    """Display a value: integers exactly, other values with at most 2 decimals."""
    if v.denominator == 1:
        return f"{v.numerator:,}" if commas else str(v.numerator)
    s = f"{float(v):,.2f}" if commas else f"{float(v):.2f}"
    return s.rstrip("0").rstrip(".")


@dataclass(frozen=True)
class _Item:
    gold: Fraction
    difficulty: float
    numbers: tuple[Fraction, ...]
    has_dollar: bool
    unit: str


def _make_item(question: str, gold: object) -> _Item:
    value = _parse_number(gold)
    if value is None:  # pseudo-gold for questions outside the answer key
        u, v = _uniform("pseudo_gold", question), _uniform("pseudo_gold_value", question)
        value = Fraction(1000 + int(v * 99000)) if u < 0.15 else Fraction(2 + int(v * 998))
    numbers: list[Fraction] = []
    for m in _NUMBER_RE.finditer(question):
        n = _parse_number(m.group())
        if n is not None and n not in numbers:
            numbers.append(n)
    has_dollar = "$" in question
    unit = "dollars" if has_dollar else "units"
    if not has_dollar:
        m = _HOW_MANY_RE.search(question)
        if m and m.group(1).lower() not in _UNIT_STOP:
            unit = m.group(1).lower()
    return _Item(
        gold=value,
        difficulty=_normal("difficulty", question) + DIFFICULTY_SHIFT,
        numbers=tuple(numbers[:8]),
        has_dollar=has_dollar,
        unit=unit,
    )


def _wrong_value(item: _Item, st: _Stream) -> Fraction:
    """A wrong final value: an intermediate quantity from the question, or the gold plus an error."""
    gold = item.gold
    if st.uniform() < 0.35:
        cands = [n for n in item.numbers if n != gold]
        if cands:
            return st.choice(cands)
    if gold != 0 and gold.denominator == 1 and abs(gold) >= 50 and st.uniform() < 0.3:
        return gold * 2 if (gold.numerator % 2 or st.uniform() < 0.5) else gold / 2
    off = st.choice((1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20))
    value = gold - off if st.uniform() < 0.5 else gold + off
    if value < 0 <= gold:
        value = gold + off
    return value


# --------------------------------------------------------------------------- text construction


def _calc_line(item: _Item, prev: Fraction | None, st: _Stream) -> tuple[str, Fraction]:
    pool = item.numbers or (Fraction(2), Fraction(3))
    a = prev if (prev is not None and st.uniform() < 0.5) else st.choice(pool)
    b = st.choice(pool)
    op = st.choice(("+", "-", "*", "/"))
    if op == "*" and abs(a * b) > 10**7:
        op = "+"
    if op == "/" and (b in (0, 1) or (a / b).denominator != 1):
        op = "+"
    if op == "-" and a < b:
        a, b = b, a
    r = a + b if op == "+" else a - b if op == "-" else a * b if op == "*" else a / b
    line = st.choice(_CALC_TEMPLATES[op]).format(a=_fmt(a), b=_fmt(b), r=_fmt(r))
    return line, r


def _reasoning(item: _Item, feats: PromptFeatures, value: Fraction, st: _Stream) -> list[str]:
    """3-8 short pseudo-reasoning lines reusing numbers from the question, ending with the value."""
    if feats.brief:
        n_lines = 3
    elif feats.step:
        n_lines = 5 + st.below(4)
    else:
        n_lines = 3 + st.below(4)
    lines: list[str] = []
    if not feats.brief and st.uniform() < 0.6:
        lines.append(st.choice(_OPENERS))
    prev: Fraction | None = None
    while len(lines) < n_lines - 1:
        line, prev = _calc_line(item, prev, st)
        lines.append(line)
    lines.append(st.choice(_SUMMARIES).format(v=_fmt(value)))
    return lines


def _final_line(fmt: str, value: Fraction, item: _Item, st: _Stream) -> str:
    v = _fmt(value)
    if fmt == "box":
        return st.choice(_BOX_LINES).format(box=f"\\boxed{{{v}}}")
    if fmt == "comma":
        return st.choice(_BOX_LINES).format(box=f"\\boxed{{{_fmt(value, commas=True)}}}")
    if fmt == "currency":
        return st.choice(_BOX_LINES).format(box=f"\\boxed{{\\${v}}}")
    if fmt == "units":
        return st.choice(_BOX_LINES).format(box=f"\\boxed{{{v} \\text{{ {item.unit}}}}}")
    if fmt == "answer_is":
        return f"The answer is {v}."
    if fmt == "bold":
        return st.choice(("**{v}**", "Final result: **{v}**")).format(v=v)
    if fmt == "hash":
        return f"#### {v}"
    raise ValueError(f"unknown format {fmt!r}")


def _cut_words(text: str, k: int) -> str:
    """``text`` up to and including its ``k``-th whitespace-separated word (line breaks kept)."""
    for i, m in enumerate(re.finditer(r"\S+", text)):
        if i == k - 1:
            return text[: m.end()]
    return text


def _truncated(item: _Item, st: _Stream, max_new_tokens: int) -> str:
    """Looping reasoning cut at the token cap, mid-line, with no final-answer marker."""
    budget = max(1, int(max_new_tokens / TOKENS_PER_WORD))
    lines = [st.choice(_OPENERS)]
    n_words, prev = len(lines[0].split()), None
    while n_words <= budget:
        if st.uniform() < 0.2:
            line = st.choice(_LOOP_LINES)
        else:
            line, prev = _calc_line(item, prev, st)
        lines.append(line)
        n_words += len(line.split())
    return _cut_words("\n".join(lines), budget - st.below(3) if budget > 3 else budget)


def _format_uniforms(system: str, question: str, skey: tuple) -> list[float]:
    """Five correlated uniforms (truncation, family, units, currency, comma) for the format draw.

    Gaussian copula: a question-level component shared by every prompt (formats depend mostly on the
    question) plus a prompt x question component, which also carries the seed under sampling.
    """
    a, b = math.sqrt(FORMAT_ITEM_CORRELATION), math.sqrt(1.0 - FORMAT_ITEM_CORRELATION)
    return [
        _NORMAL.cdf(
            a * _normal("format_item", question, k) + b * _normal("format", system, question, *skey, k)
        )
        for k in range(5)
    ]


def _pick(pairs: Sequence[tuple[str, float]], u: float) -> str:
    """Weighted categorical choice driven by one uniform ``u``."""
    x = u * sum(w for _, w in pairs)
    for key, w in pairs:
        x -= w
        if x < 0:
            return key
    return pairs[-1][0]


def _choose_format(
    feats: PromptFeatures, item: _Item, value: Fraction, solved: bool, temperature: float, u: Sequence[float]
) -> str:
    """Categorical final-answer format given the prompt's format instructions (seeded under sampling)."""
    heat = temperature / REFERENCE_TEMPERATURE  # 0 for greedy
    noise = 1.0 + _FORMAT_HEAT_NOISE * heat
    p_trunc = _TRUNC_BASE + (0.0 if solved else _TRUNC_UNSOLVED) + (_TRUNC_STEP if feats.step else 0.0)
    p_trunc = (p_trunc + _TRUNC_HEAT * heat) * (0.3 if feats.brief else 1.0)
    if u[0] < p_trunc:
        return "truncated"
    if feats.boxed:
        family = _FAMILY_PLAIN_PROMPT if feats.plain else _FAMILY_BOXED_PROMPT
        family = tuple((k, w if k in ("boxed", "hash") else w * noise) for k, w in family)
    else:
        family = _FAMILY_NO_BOX_PROMPT
    fam = _pick(family, u[1])
    if fam != "boxed":
        return fam
    if feats.plain:
        p_units, p_cur, p_comma = (0.2 if feats.units_format else 0.005), 0.01, 0.03
    else:
        p_units, p_cur, p_comma = (0.45 if feats.units_format else 0.03), 0.18, 0.5
    if u[2] < p_units * noise:
        return "units"
    if item.has_dollar and u[3] < p_cur * noise:
        return "currency"
    if abs(value) >= 1000 and u[4] < p_comma * noise:
        return "comma"
    return "box"


@dataclass(frozen=True)
class _Trace:
    solved: bool
    flipped: bool
    diverged: bool
    value: Fraction
    fmt: str
    quality: float
    difficulty: float
    margin: float


# --------------------------------------------------------------------------- proposer


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _apply_edit(sents: list[str], st: _Stream) -> list[str]:
    """One seeded edit: add a snippet, replace a non-format sentence with one, or remove one."""
    present = {_norm(s) for s in sents}
    avail = [s for s in SNIPPETS if _norm(s.text) not in present]
    removable = [i for i, s in enumerate(sents) if not _BOXED_RE.search(s)] if len(sents) > 1 else []
    long_prompt = sum(len(s.split()) for s in sents) > 150
    ops = _EDIT_OPS_LONG if long_prompt else _EDIT_OPS
    if len(removable) <= 1:  # rarely strip the last non-format sentence
        ops = tuple((k, w * 0.25 if k == "remove" else w) for k, w in ops)
    op = st.weighted(ops)
    if op != "remove" and not avail:
        op = "remove"
    if op != "add" and not removable:
        op = "add"
    out = list(sents)
    if op == "remove":
        del out[st.choice(removable)]
        return out
    if not avail:
        return out
    snippet = st.weighted([(s.text, s.weight) for s in avail])
    if op == "replace":
        out[st.choice(removable)] = snippet
    else:
        out.insert(1 + st.below(len(out)), snippet)
    return out


def _edit_prompt(prompt: str, st: _Stream) -> str:
    """1-2 seeded edits; a sequence that restores the incumbent (add then remove) gets another edit."""
    original = _sentences(prompt)
    sents = original
    for _ in range(1 if st.uniform() < 0.6 else 2):
        sents = _apply_edit(sents, st)
    for _ in range(3):
        if [_norm(s) for s in sents] != [_norm(s) for s in original]:
            break
        sents = _apply_edit(sents, st)
    return " ".join(sents)


def _shown_golds(meta_prompt: str) -> list[str]:
    """Numbers with >= 3 digits among the gold answers shown in the meta-prompt."""
    out = []
    for m in _GOLD_BLOCK_RE.finditer(meta_prompt):
        for n in _NUMBER_RE.findall(m.group(1)):
            if sum(c.isdigit() for c in n) >= 3 and n not in out:
                out.append(n)
    return out


def _malformed(candidate: str, meta_prompt: str, preamble: str, st: _Stream) -> tuple[str, str]:
    """(text, kind) of a malformed proposal; ``copies_gold`` falls back to ``no_boxed`` without large golds."""
    kind = st.choice(MALFORMED_KINDS)
    if kind == "copies_gold":
        golds = _shown_golds(meta_prompt)
        if golds:
            g = st.choice(golds)
            body = f"{candidate} For example, a final answer of {g} is written as \\boxed{{{g}}}."
            return f"{preamble}\n<prompt>\n{body}\n</prompt>", kind
        kind = "no_boxed"
    if kind == "missing_tags":
        return f"{preamble}\n\n{candidate}", kind
    if kind == "empty_tags":
        return f"{preamble}\n<prompt>\n</prompt>", kind
    if kind == "nested":
        return f"{preamble}\n<prompt>\n<prompt>\n{candidate}\n</prompt>\n</prompt>", kind
    kept = [s for s in _sentences(candidate) if not _BOXED_RE.search(s)] or [_NO_BOX_FALLBACK]
    return f"{preamble}\n<prompt>\n{' '.join(kept)}\n</prompt>", kind


# --------------------------------------------------------------------------- backend


def _n_tokens(text: str) -> int:
    return math.ceil(TOKENS_PER_WORD * len(text.split()))


def answer_key_from_jsonl(path: str | Path) -> dict[str, str]:
    """``{question: gold}`` from a GSM8K snapshot file (``gold`` field, else the ``####`` of ``answer``)."""
    key: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        gold = row.get("gold")
        if gold is None:
            gold = str(row["answer"]).split("####")[-1].strip().replace(",", "")
        key[row["question"]] = str(gold)
    return key


class MockBackend(Backend):
    """Deterministic synthetic backend. Every run it produces is ``synthetic`` (never experimental data)."""

    kind = "mock"
    synthetic = True

    def __init__(
        self,
        model_id: str,
        model_revision: str,
        cfg: MockSection | None = None,
        answer_key: Mapping[str, str] | None = None,
    ) -> None:
        """``answer_key`` maps the exact user message (the question) to its gold answer (numeric string;
        ``Fraction``/int values are accepted too). Questions outside it get a hashed pseudo-gold."""
        super().__init__(model_id, model_revision)
        self.cfg = cfg if cfg is not None else MockSection()
        self._answer_key: dict[str, str] = {str(k): str(v) for k, v in (answer_key or {}).items()}
        self._items: dict[str, _Item] = {}
        self._quality: dict[str, float] = {}

    # -- protocol -------------------------------------------------------------------------
    def engine_info(self) -> dict:
        info: dict[str, object] = {"kind": self.kind, "version": MOCK_VERSION}
        info.update(self.cfg.model_dump(mode="json"))  # base_quality, sample_sd, greedy_flip_rate, ...
        info["answer_key"] = sha256_json(sorted(self._answer_key.items()))[:16] if self._answer_key else None
        return info

    def render(self, system: str, user: str) -> str:
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def generate(self, reqs: Sequence[GenRequest]) -> list[GenResult]:
        return [self._generate_one(r) for r in reqs]

    # -- introspection (mock only; for tests and the synthetic demo) -----------------------
    def quality(self, system: str) -> float:
        """Latent quality ``q(p)`` of a system prompt under this config."""
        q = self._quality.get(system)
        if q is None:
            q = self._quality[system] = prompt_quality(system, self.cfg.base_quality)
        return q

    def gold_for(self, user: str) -> str:
        """The gold answer the mock uses for a question (answer key, else a hashed pseudo-gold)."""
        return _fmt(self._item(user).gold)

    def explain(self, req: GenRequest) -> dict[str, object]:
        """Latent variables behind a generation (answer mode: solved, format, flip, ...; proposer mode:
        whether the proposal is malformed and how)."""
        if req.system == PROPOSER_SYSTEM:
            _, _, kind, found = self._proposal(req)
            return {"mode": "proposer", "malformed": kind, "incumbent_found": found}
        t = self._trace(req)
        return {
            "mode": "answer",
            "solved": t.solved,
            "flipped": t.flipped,
            "diverged": t.diverged,
            "value": _fmt(t.value),
            "gold": self.gold_for(req.user),
            "format": t.fmt,
            "quality": t.quality,
            "difficulty": t.difficulty,
            "margin": t.margin,
        }

    # -- internals ------------------------------------------------------------------------
    def _item(self, user: str) -> _Item:
        item = self._items.get(user)
        if item is None:
            gold = self._answer_key.get(user)
            if gold is None:
                gold = self._answer_key.get(user.strip())
            item = self._items[user] = _make_item(user, gold)
        return item

    def _trace(self, req: GenRequest) -> _Trace:
        dec, cfg = req.decoding, self.cfg
        skey = _decode_key(dec, req.seed)
        item, feats, q = self._item(req.user), prompt_features(req.system), self.quality(req.system)
        margin = q - item.difficulty + INTERACTION_SCALE * _normal("interaction", req.system, req.user)
        if not dec.is_greedy:
            sd = cfg.sample_sd * dec.temperature / REFERENCE_TEMPERATURE
            margin += sd * _normal("eps", req.system, req.user, *skey)
        solved, flipped, diverged = margin > 0, False, False
        if req.nonce is not None and dec.is_greedy and cfg.greedy_flip_rate > 0:
            u = _uniform("flip", req.nonce, req.system, req.user)
            if u < cfg.greedy_flip_rate:
                solved, flipped = not solved, True
            elif u < min(1.0, 2.0 * cfg.greedy_flip_rate):
                diverged = True
        if solved:
            value = item.gold
        else:
            salt = ("flip", req.nonce) if flipped else ()
            value = _wrong_value(item, _Stream("value", req.system, req.user, *skey, *salt))
        temperature = 0.0 if dec.is_greedy else dec.temperature
        uniforms = _format_uniforms(req.system, req.user, skey)
        fmt = _choose_format(feats, item, value, solved, temperature, uniforms)
        return _Trace(solved, flipped, diverged, value, fmt, q, item.difficulty, margin)

    def _answer(self, req: GenRequest) -> tuple[str, str]:
        t = self._trace(req)
        salt = ("rerun", req.nonce) if (t.flipped or t.diverged) else ()
        st = _Stream("text", req.system, req.user, *_decode_key(req.decoding, req.seed), *salt)
        item = self._item(req.user)
        if t.fmt == "truncated":
            return _truncated(item, st, req.decoding.max_new_tokens), "length"
        lines = _reasoning(item, prompt_features(req.system), t.value, st)
        lines.append(_final_line(t.fmt, t.value, item, st))
        return "\n".join(lines), "stop"

    def _proposal(self, req: GenRequest) -> tuple[str, str, str | None, bool]:
        """(text, finish_reason, malformed kind or None, incumbent found in the meta-prompt)."""
        st = _Stream("proposer", req.system, req.user, *_decode_key(req.decoding, req.seed))
        found = (extract_current_prompt(req.user) or "").strip()
        malformed = st.uniform() < self.cfg.malformed_proposal_rate
        candidate = _edit_prompt(found or _DEFAULT_PROMPT, st)
        preamble = st.choice(_PREAMBLES)
        if malformed:
            text, kind = _malformed(candidate, req.user, preamble, st)
            return text, "stop", kind, bool(found)
        return f"{preamble}\n<prompt>\n{candidate}\n</prompt>", "stop", None, bool(found)

    def _generate_one(self, req: GenRequest) -> GenResult:
        if req.system == PROPOSER_SYSTEM:
            text, finish, _, _ = self._proposal(req)
        else:
            text, finish = self._answer(req)
        cap = req.decoding.max_new_tokens
        n_completion = _n_tokens(text)
        if finish == "length":
            n_completion = cap
        elif n_completion > cap:  # honour max_new_tokens: cut and report truncation
            text, finish, n_completion = _cut_words(text, max(1, int(cap / TOKENS_PER_WORD))), "length", cap
        n_prompt = math.ceil(TOKENS_PER_WORD * (len(req.system.split()) + len(req.user.split())))
        latency = round(2.0 + 0.02 * n_completion + 0.002 * n_prompt, 3)
        return GenResult(
            text=text,
            finish_reason=finish,
            n_prompt_tokens=n_prompt,
            n_completion_tokens=n_completion,
            latency_ms=latency,
        )


__all__ = [
    "FORMATS",
    "MALFORMED_KINDS",
    "MOCK_VERSION",
    "QUALITY_WEIGHTS",
    "SNIPPETS",
    "STRICT_FORMATS",
    "MockBackend",
    "PromptFeatures",
    "Snippet",
    "answer_key_from_jsonl",
    "prompt_features",
    "prompt_quality",
]
