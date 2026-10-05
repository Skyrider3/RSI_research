"""Shared, FROZEN building blocks of the answer extractors (a research artifact).

Everything that decides whether a response counts as correct lives in this module or in a version module
(``v1.py``, ``v2.py``). Their source bytes are hashed into ``extractor_hash`` and every stored score
references that hash, so this file must not change behaviour: a different rule becomes a NEW version
(``v3.py``) instead. (The semantics below were finalised before any real experiment run.)

Marker selection is shared on purpose: both extractors first call :func:`select_marker` (which applies the
``####`` qualification rule below) and read the same payload (:func:`marker_payload`), and
:func:`parse_lenient` returns ``Fraction(Decimal(s))`` for every string ``s`` that v1 accepts. Hence
v1-correct implies v2-correct (monotonicity), which the analyses rely on.

A ``####`` is an answer marker only when the rest of its line, stripped, starts like a number
(:data:`HASH_ANSWER_RE`: a digit or ``.digit`` after an optional sign / dollar / minus, e.g. ``#### 42``,
``#### -3``, ``#### $1,234``, ``#### .5``, ``#### 4.``) and is not a numbered heading
(:data:`NUMBERED_HEADING_RE`: ``#### 4. Verification``, ``#### 2) Check``). Markdown headings
(``#### Step 4: Verify``, ``#### Final Answer``), numbered headings, numberless lines (``#### ...``,
``#### .``, ``#### -.``) and empty ``####`` lines are ignored by BOTH extractors (:func:`is_hash_answer`), and
the LAST qualifying occurrence is used.
"""

from __future__ import annotations

import bisect
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction

METHODS: tuple[str, ...] = ("boxed", "hash", "fbox", "answer_phrase", "bold", "last_number", "none")

BOX_RE = re.compile(r"\\boxed\s*\{")
FBOX_RE = re.compile(r"\\fbox\s*\{")
HASH_MARK = "####"
# A "####" qualifies as an answer marker iff, for c = <rest of its line, stripped>, HASH_ANSWER_RE.match(c)
# and not NUMBERED_HEADING_RE.match(c) (see is_hash_answer).
# HASH_ANSWER_RE: optional sign, optional (escaped) dollar, optional minus, then a digit or a '.' followed by
# a digit (whitespace allowed in between; empty content, "...", "." and "-." do not qualify, ".5" does).
# Written without adjacent "\s*" runs, so long blank runs cannot cause polynomial backtracking; it accepts
# exactly the strings of the spec pattern r"[-+\u2212]?\s*(?:\\?\$)?\s*[-\u2212]?\s*(?:\d|\.\d)" (an
# equivalence test pins this).
HASH_ANSWER_RE = re.compile(r"[-+\u2212]?\s*(?:\\?\$\s*)?(?:[-\u2212]\s*)?(?:\d|\.\d)")
# A numbered markdown heading ("4. Verification", "2) Check the answer"): never an answer marker. "42",
# "42.", "3.5" and "4." are not headings (the '.'/')' must be followed by whitespace and a letter).
NUMBERED_HEADING_RE = re.compile(r"\d+[.)]\s+[A-Za-z]")
_SPACE_RE = re.compile(r"\s*")  # same character class as str.strip() (Unicode whitespace)
# A backslash escapes the next character (so "\{" and "\}" are literal); bare braces nest.
_BRACE_TOKEN_RE = re.compile(r"\\.|[{}]", re.S)

# Lenient number grammar: thousands separators, decimals, leading-dot decimals; no letters/dots before.
NUM_RE = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|(?<![\w\d])-?\.\d+")
_UNWRAP_RE = re.compile(r"\\(?:textbf|textrm|text|mathrm|mathbf|mbox)\s*\{([^{}]*)\}")
_FRAC_RE = re.compile(r"^(-?)\\[dt]?frac\{(-?[\d.,]+)\}\{(-?[\d.,]+)\}$")
_RATIO_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)$")
# A leading minus separated from the first number by whitespace ("- 7", "-  .5"); parse_lenient joins them.
_SPACED_SIGN_RE = re.compile(r"-\s+(?=\d|\.\d)")
_DROP_SPACING = ("\\,", "\\!", "\\;", "\\:", "~")
_DROP_TOKENS = ("\\$", "$", "\\%", "%", "\\left", "\\right", "^{\\circ}", "^\\circ")
UNICODE_MINUS = "\u2212"

Span = tuple[int, int]
Marker = tuple[str, str, Span]  # (kind "boxed" | "hash", content, span of content in the original text)


@dataclass(frozen=True)
class Extraction:
    """Result of one extractor on one response.

    ``span``/``content``: the substring of the ORIGINAL text that the rule parsed (``text[s:e] == content``
    whenever ``span`` is set). v1 also reports the rejected marker content when its value is ``None``.
    """

    value: Fraction | None
    method: str  # one of METHODS
    span: Span | None = None
    content: str | None = None

    @property
    def extracted(self) -> str | None:
        """Canonical string of the value (what ``scores.extracted`` stores), or ``None``."""
        return None if self.value is None else canonical(self.value)


NO_EXTRACTION = Extraction(None, "none", None, None)


# --------------------------------------------------------------------------- numbers and gold


def to_fraction(s: str) -> Fraction | None:
    """Exact value of a decimal string (``Fraction(Decimal(s))``), or ``None`` if it does not parse."""
    try:
        return Fraction(Decimal(s))
    except (InvalidOperation, ValueError, OverflowError):
        return None


def _digits(n: int) -> str:
    """Decimal digits of a non-negative int; via Decimal, so exempt from the int->str digit limit."""
    return str(Decimal(n))


def canonical(value: Fraction | int) -> str:
    """Canonical string: "18", "-3", "0.5", "1234.5" for terminating decimals, "p/q" otherwise."""
    v = Fraction(value)
    sign = "-" if v < 0 else ""
    num, den = abs(v.numerator), v.denominator
    if den == 1:
        return f"{sign}{_digits(num)}"
    twos = fives = 0
    d = den
    while d % 2 == 0:
        d //= 2
        twos += 1
    while d % 5 == 0:
        d //= 5
        fives += 1
    if d != 1:
        return f"{sign}{_digits(num)}/{_digits(den)}"
    k = max(twos, fives)  # minimal k with value * 10**k integral -> no trailing zeros
    digits = _digits(num * 10**k // den).rjust(k + 1, "0")
    return f"{sign}{digits[:-k]}.{digits[-k:]}"


def gold_value(gold: str | Fraction | int) -> Fraction:
    """Gold answer as an exact Fraction. Accepts ints, Fractions, decimal strings and canonical "p/q"."""
    if isinstance(gold, Fraction):
        return gold
    if isinstance(gold, int) and not isinstance(gold, bool):
        return Fraction(gold)
    s = str(gold).replace(",", "").strip()
    if "/" in s:  # canonical() emits "p/q" for non-terminating values
        num, den = s.split("/", 1)
        return Fraction(Decimal(num.strip())) / Fraction(Decimal(den.strip()))
    return Fraction(Decimal(s))


def parse_gold(answer_text: str) -> Fraction:
    """GSM8K gold: the number after the last ``####`` of the reference solution (commas removed)."""
    return Fraction(Decimal(answer_text.split("####")[-1].strip().replace(",", "")))


def is_correct(ex: Extraction, gold: str | Fraction | int) -> bool:
    """Exact Fraction equality ("5.00" equals 5; 4.99 does not)."""
    return ex.value is not None and ex.value == gold_value(gold)


# --------------------------------------------------------------------------- markers


def match_brace(text: str, open_idx: int) -> int | None:
    """Index of the ``}`` closing the ``{`` at ``open_idx`` (balanced scan), or ``None`` if unbalanced."""
    depth = 0
    for m in _BRACE_TOKEN_RE.finditer(text, open_idx):
        tok = m.group()
        if tok == "{":
            depth += 1
        elif tok == "}":
            depth -= 1
            if depth == 0:
                return m.start()
    return None


def brace_pairs(text: str) -> dict[int, int]:
    """``{open_idx: close_idx}`` for every balanced brace, in one left-to-right pass.

    Same result as ``match_brace`` from each ``{`` (stray ``}`` with nothing open are ignored), in linear time.
    """
    pairs: dict[int, int] = {}
    stack: list[int] = []
    for m in _BRACE_TOKEN_RE.finditer(text):
        tok = m.group()
        if tok == "{":
            stack.append(m.start())
        elif tok == "}" and stack:
            pairs[stack.pop()] = m.start()
    return pairs


def _scan_braced(text: str, pattern: re.Pattern[str]) -> list[tuple[int, str, int, int]]:
    """(marker_start, raw content, content_start, content_end) for every balanced ``\\cmd{...}``."""
    markers = list(pattern.finditer(text))
    if not markers:
        return []
    pairs = brace_pairs(text)
    out = []
    for m in markers:
        open_idx = m.end() - 1  # never escaped: the pattern puts a letter or whitespace before "{"
        close = pairs.get(open_idx)
        if close is not None:  # unbalanced (e.g. truncated) boxes are dropped
            out.append((m.start(), text[open_idx + 1 : close], open_idx + 1, close))
    return out


def find_boxes(text: str) -> list[tuple[str, int, int]]:
    """Every balanced ``\\boxed{...}`` in order of appearance: (raw content, start, end) of the content."""
    return [(c, s, e) for _, c, s, e in _scan_braced(text, BOX_RE)]


def find_fboxes(text: str) -> list[tuple[str, int, int]]:
    """Every balanced ``\\fbox{...}`` in order of appearance: (raw content, start, end) of the content."""
    return [(c, s, e) for _, c, s, e in _scan_braced(text, FBOX_RE)]


def strip_span(content: str, start: int) -> tuple[str, Span]:
    """Strip whitespace from ``content`` (found at ``start``) and return it with its span in the text."""
    lead = len(content) - len(content.lstrip())
    stripped = content.strip()
    return stripped, (start + lead, start + lead + len(stripped))


def _qualifies(text: str, first: int, line_end: int) -> bool:
    """The shared ``####`` rule on ``text[first:line_end]`` (``first`` = first non-blank character).

    Both patterns end in a non-blank token (a digit / a letter), so testing the unstripped tail of the line
    is the same as testing the stripped content.
    """
    return (
        HASH_ANSWER_RE.match(text, first, line_end) is not None
        and NUMBERED_HEADING_RE.match(text, first, line_end) is None
    )


def is_hash_answer(content: str) -> bool:
    """Whether the rest of a ``####`` line qualifies as an answer: stripped, it starts like a number
    (:data:`HASH_ANSWER_RE`) and is not a numbered heading (:data:`NUMBERED_HEADING_RE`)."""
    content = content.strip()
    return _qualifies(content, 0, len(content))


def _last_hash(text: str) -> tuple[int, str, Span] | None:
    """LAST qualifying ``####`` (occurrences scanned from the end): (marker start, stripped content, span).

    Equivalent to testing ``is_hash_answer(text[pos + 4 : end_of_line])`` for every occurrence ``pos``
    (overlapping ones included, so "##### 42" reads " 42"), but linear: each line's bounds are located once,
    and the qualification test reads only the first characters of the rest of the line (both patterns stop
    at the next '#', so overlapping tests on one line never re-read each other's text).
    """
    end = len(text)
    line_start, line_end = end + 1, end
    while (pos := text.rfind(HASH_MARK, 0, end)) >= 0:
        start = pos + len(HASH_MARK)
        if pos < line_start:  # first (= last) occurrence on a new line: locate the line once
            nl = text.find("\n", start)
            line_end = len(text) if nl < 0 else nl
            line_start = text.rfind("\n", 0, pos) + 1
        first = _SPACE_RE.match(text, start, line_end).end()  # endpos: never crosses the newline
        if _qualifies(text, first, line_end):
            content, span = strip_span(text[start:line_end], start)
            return pos, content, span
        end = pos + len(HASH_MARK) - 1  # next occurrence starts before pos (overlaps allowed)
    return None


def last_hash_line(text: str) -> tuple[str, int, int] | None:
    """The rest of the line after the LAST QUALIFYING ``####`` (stripped) with its span, or ``None``.

    Headings (``#### Step 4: Verify``, ``#### Final Answer``, ``#### 4. Verification``), numberless lines
    (``#### ...``) and empty ``####`` lines never qualify.
    """
    h = _last_hash(text)
    if h is None:
        return None
    _, content, (start, end) = h
    return content, start, end


def _last_box(text: str) -> tuple[int, str, Span] | None:
    boxes = _scan_braced(text, BOX_RE)
    if not boxes:
        return None
    pos, raw, start, _ = boxes[-1]
    content, span = strip_span(raw, start)
    return pos, content, span


def select_marker(text: str) -> Marker | None:
    """Whichever of (last balanced box, last QUALIFYING ``####`` line) starts later; content stripped.

    Both v1 and v2 call this first, which guarantees monotonicity (v1-correct implies v2-correct).
    """
    box, hsh = _last_box(text), _last_hash(text)
    if box is not None and (hsh is None or box[0] > hsh[0]):
        return "boxed", box[1], box[2]
    if hsh is not None:
        return "hash", hsh[1], hsh[2]
    return None


def other_marker(text: str, kind: str) -> Marker | None:
    """The marker of the other kind (last box if ``kind == "hash"``, else the last qualifying ``####`` line)."""
    found = _last_box(text) if kind == "hash" else _last_hash(text)
    if found is None:
        return None
    return ("boxed" if kind == "hash" else "hash"), found[1], found[2]


def marker_payload(kind: str, content: str, span: Span) -> tuple[str, Span]:
    """Content a marker contributes: the box content, or the ``####`` line minus ONE sentence-final '.'."""
    if kind == "hash" and content.endswith("."):
        content = content[:-1].rstrip()
        span = (span[0], span[0] + len(content))
    return content, span


# --------------------------------------------------------------------------- lenient parsing


def last_match(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    """Last non-overlapping match of ``pattern`` (``finditer`` order), or ``None``."""
    found = None
    for m in pattern.finditer(text):
        found = m
    return found


def normalize(s: str) -> str:
    """NFKC normalisation with the unicode minus sign (U+2212) mapped to '-'."""
    return unicodedata.normalize("NFKC", s).replace(UNICODE_MINUS, "-")


def strip_outer_braces(s: str) -> str:
    """Strip whitespace and balanced outer braces repeatedly ("{{42}}" -> "42", "{1}{2}" unchanged)."""
    s = s.strip()
    while len(s) >= 2 and s[0] == "{" and s[-1] == "}" and match_brace(s, 0) == len(s) - 1:
        s = s[1:-1].strip()
    return s


def _quotient(num: str, den: str, negative: bool) -> Fraction | None:
    a, b = to_fraction(num.replace(",", "")), to_fraction(den.replace(",", ""))
    if a is None or b is None or b == 0:
        return None
    q = a / b
    return -q if negative else q


def parse_lenient(s: str) -> Fraction | None:
    """Lenient numeric parse of a short answer string (box content, ``####`` payload, answer phrase, bold).

    Normalisation, in order: NFKC + unicode minus -> '-'; unwrap ``\\text{}``-like wrappers (3 levels); drop
    LaTeX spacing, dollars, percents, ``\\left``/``\\right`` and degree marks (``{,}`` -> ','); strip
    whitespace and balanced outer braces; keep only what follows the LAST '=' ("12 - 7 = 5" -> "5"). Then a
    leading minus separated from the first number by whitespace is joined to it ("- 7", "$ - 7",
    "x = - 7" -> -7; "3 - 4" stays 3). Value: ``\\frac{a}{b}`` / ``a/b`` as a whole, else the FIRST number.
    """
    s = normalize(s)
    for _ in range(3):
        s, n = _UNWRAP_RE.subn(lambda m: m.group(1), s)
        if n == 0:
            break
    for tok in _DROP_SPACING:
        s = s.replace(tok, "")
    s = s.replace("{,}", ",")
    for tok in _DROP_TOKENS:
        s = s.replace(tok, "")
    s = strip_outer_braces(s)
    if "=" in s:
        s = s.rsplit("=", 1)[1].strip()
    m = _SPACED_SIGN_RE.match(s)
    if m:  # never fires on a v1-valid string (no whitespace), so the monotonicity lemma is untouched
        s = "-" + s[m.end() :]
    m = _FRAC_RE.match(s)
    if m:
        return _quotient(m.group(2), m.group(3), negative=m.group(1) == "-")
    m = _RATIO_RE.match(s)
    if m:
        return _quotient(m.group(1), m.group(2), negative=False)
    m = NUM_RE.search(s)
    if m:
        return to_fraction(m.group().replace(",", ""))
    return None


def last_number(text: str) -> tuple[Fraction, Span | None, str] | None:
    """Last ``NUM_RE`` match in the normalised text: (value, span in the ORIGINAL text, matched string).

    The span is exact whenever per-character normalisation reproduces the whole-text normalisation (always
    for ASCII, full-width digits, ligatures, ...); otherwise (combining sequences) it is ``None``.
    """
    norm = normalize(text)
    last = last_match(NUM_RE, norm)
    if last is None:
        return None
    value = to_fraction(last.group().replace(",", ""))
    if value is None:  # pragma: no cover - NUM_RE matches always parse
        return None
    a, b = last.span()
    if norm == text:
        return value, (a, b), text[a:b]
    pieces = [normalize(ch) for ch in text]
    if "".join(pieces) != norm:
        return value, None, last.group()
    starts = [0]
    for p in pieces:
        starts.append(starts[-1] + len(p))
    s = bisect.bisect_right(starts, a) - 1
    e = bisect.bisect_left(starts, b)
    return value, (s, e), text[s:e]
