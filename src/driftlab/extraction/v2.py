"""Extractor v2 (lenient, FROZEN). Stages in order; the first value that parses wins:

1. the ``select_marker`` payload (shared with v1) via ``parse_lenient``; then the OTHER marker
   (the last box if the ``####`` line was selected, or vice versa)            -> "boxed" / "hash"
2. the last balanced ``\\fbox{...}``                                            -> "fbox"
3. the last "(final) answer is / : / =" phrase                                -> "answer_phrase"
4. the last ``**bold**`` / ``__bold__`` span whose stripped content is
   number-like (``BOLD_NUMERIC_RE``)                                          -> "bold"
5. the last number in the NFKC-normalised response                            -> "last_number"

Only QUALIFYING ``####`` lines are markers (shared rule, ``common.is_hash_answer``): a heading such as
``#### Step 3``, ``#### Final Answer`` or ``#### 4. Verification`` and a numberless line such as ``#### ...``
are not markers, so the later stages decide. ``parse_lenient`` keeps a leading sign separated from the
number by whitespace (``#### - 7``, ``\\boxed{$ - 7}``, ``**- 7**`` -> -7). The bold stage accepts
``**42**``, ``**$1,234**``, ``**18 dollars**``, ``**25%**``, ``**x = 18**``, ``**3.5 hours.**``, ``**-7**``
and skips step headers / equations such as ``**Step 3: Add the parts**`` or ``**Total = 5 + 13 = 18**``.

Because stage 1 reads exactly the payload v1 reads and ``parse_lenient`` agrees with v1 on every string v1
accepts, v1-correct implies v2-correct.
"""

from __future__ import annotations

import re

from driftlab.extraction.common import (
    NO_EXTRACTION,
    Extraction,
    find_fboxes,
    last_match,
    last_number,
    marker_payload,
    other_marker,
    parse_lenient,
    select_marker,
    strip_span,
)

ANSWER_PHRASE_RE = re.compile(
    r"(?i)\b(?:final\s+answer|the\s+answer|answer)\b[^\n\d-]{0,20}?(?:is|:|=)\s*(?P<rest>[^\n]{0,80})"
)
BOLD_RES = (re.compile(r"\*\*([^*\n]{1,40})\*\*"), re.compile(r"__([^_\n]{1,40})__"))
# Number-like bold content: optional "x =", sign, (escaped) dollar, minus; a number (thousands commas,
# decimals, leading-dot decimals); optional "%" or one/two unit words; optional final '.'/'!'. Written
# without adjacent "\s*" runs; it accepts exactly the strings of the spec pattern
# r"^\s*(?:[A-Za-z]\s*=\s*)?[-+\u2212]?\s*(?:\\?\$)?\s*[-\u2212]?\s*(?:\d[\d,]*(?:\.\d+)?|\.\d+)\s*"
# r"(?:\\?%|[A-Za-z]+(?:\s+[A-Za-z]+)?)?\s*[.!]?\s*$" (an equivalence test pins this).
BOLD_NUMERIC_RE = re.compile(
    r"^\s*(?:[A-Za-z]\s*=\s*)?(?:[-+\u2212]\s*)?(?:\\?\$\s*)?(?:[-\u2212]\s*)?(?:\d[\d,]*(?:\.\d+)?|\.\d+)\s*"
    r"(?:(?:\\?%|[A-Za-z]+(?:\s+[A-Za-z]+)?)\s*)?(?:[.!]\s*)?$"
)


def _from_marker(text: str) -> Extraction | None:
    first = select_marker(text)
    if first is None:
        return None
    for marker in (first, other_marker(text, first[0])):
        if marker is None:
            continue
        kind, content, span = marker
        content, span = marker_payload(kind, content, span)
        value = parse_lenient(content)
        if value is not None:
            return Extraction(value, kind, span, content)
    return None


def _from_fbox(text: str) -> Extraction | None:
    boxes = find_fboxes(text)
    if not boxes:
        return None
    raw, start, _ = boxes[-1]
    content, span = strip_span(raw, start)
    value = parse_lenient(content)
    if value is None:
        return None
    return Extraction(value, "fbox", span, content)


def _from_answer_phrase(text: str) -> Extraction | None:
    m = last_match(ANSWER_PHRASE_RE, text)
    if m is None:
        return None
    value = parse_lenient(m.group("rest"))
    if value is None:
        return None
    return Extraction(value, "answer_phrase", m.span("rest"), m.group("rest"))


def _from_bold(text: str) -> Extraction | None:
    spans = sorted((m.span(1) for rx in BOLD_RES for m in rx.finditer(text)), reverse=True)
    for start, end in spans:
        content, span = strip_span(text[start:end], start)
        if BOLD_NUMERIC_RE.match(content) is None:  # step headers, equations, prose
            continue
        value = parse_lenient(content)
        if value is not None:
            return Extraction(value, "bold", span, content)
    return None


def _from_last_number(text: str) -> Extraction | None:
    found = last_number(text)
    if found is None:
        return None
    value, span, content = found
    return Extraction(value, "last_number", span, content)


_STAGES = (_from_marker, _from_fbox, _from_answer_phrase, _from_bold, _from_last_number)


def extract(text: str) -> Extraction:
    """Lenient extraction (see the module docstring for the stage order)."""
    for stage in _STAGES:
        ex = stage(text)
        if ex is not None:
            return ex
    return NO_EXTRACTION
