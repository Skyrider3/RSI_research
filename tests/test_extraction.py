"""Golden, property and frozen-hash tests for the answer extractors (a frozen research artifact)."""

from __future__ import annotations

import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from driftlab.extraction import (
    METHODS,
    NO_EXTRACTION,
    REGISTRY,
    Extraction,
    canonical,
    extract,
    extract_v1,
    extract_v2,
    extractor_hash,
    extractor_tag,
    extractor_tags,
    find_boxes,
    frozen_mismatches,
    gold_value,
    is_correct,
    last_hash_line,
    parse_gold,
    parse_lenient,
    select_marker,
    verify_frozen,
)
from driftlab.extraction import _hash_sources as hash_sources
from driftlab.extraction.common import BOX_RE, match_brace

REPO = Path(__file__).resolve().parents[1]
FROZEN_JSON = Path(__file__).with_name("extractors_frozen.json")
MINUS = "\u2212"

# --------------------------------------------------------------------------- golden cases
# (text, v1 value, v2 value, v1 method, v2 method); values are Fraction() strings or None.
GOLDEN: list[tuple[str, str | None, str | None, str, str]] = [
    # --- required table ------------------------------------------------------------------
    (r"\boxed{42}", "42", "42", "boxed", "boxed"),
    (r"\boxed{ 42 }", "42", "42", "boxed", "boxed"),
    (r"\boxed{-3}", "-3", "-3", "boxed", "boxed"),
    (r"\boxed{5.00}", "5", "5", "boxed", "boxed"),
    (r"\boxed{5.}", None, "5", "none", "boxed"),
    (r"\boxed{.5}", None, "1/2", "none", "boxed"),
    (r"\boxed{1,234}", None, "1234", "none", "boxed"),
    (r"\boxed{\$1,234.50}", None, "1234.5", "none", "boxed"),
    (r"\boxed{18 \text{ dollars}}", None, "18", "none", "boxed"),
    (r"\boxed{\text{18}}", None, "18", "none", "boxed"),
    (r"\boxed{x = 18}", None, "18", "none", "boxed"),
    (r"\boxed{3 \times 6 = 18}", None, "18", "none", "boxed"),
    (r"\boxed{\dfrac{3}{4}}", None, "3/4", "none", "boxed"),
    (r"\boxed{25\%}", None, "25", "none", "boxed"),
    ("First \\boxed{5}, but on reflection the total is \\boxed{6}.", "6", "6", "boxed", "boxed"),
    ("So we get \\boxed{5}.\nDouble-checking:\n#### 7", "7", "7", "hash", "hash"),
    ("#### 7\nActually, correcting the error: \\boxed{5}", "5", "5", "boxed", "boxed"),
    ("#### 42", "42", "42", "hash", "hash"),
    ("#### 42.", "42", "42", "hash", "hash"),
    ("#### 1,234", None, "1234", "none", "hash"),
    ("#### $42", None, "42", "none", "hash"),
    ("#### 42 apples", None, "42", "none", "hash"),
    ("Step 1: so 3 + 1 gives the total \\boxed{4", None, "4", "none", "last_number"),
    ("The answer is $1,234.", None, "1234", "none", "answer_phrase"),
    ("**42**", None, "42", "none", "bold"),
    ("Final Answer: 42", None, "42", "none", "answer_phrase"),
    (f"\\boxed{{{MINUS}3}}", None, "-3", "none", "boxed"),
    ("it takes 3-4 hours", None, "4", "none", "last_number"),
    (r"\boxed{{42}}", None, "42", "none", "boxed"),
    ("We need \\boxed{}\nSo the answer is 9", None, "9", "none", "answer_phrase"),
    ("I am not able to solve this problem.", None, None, "none", "none"),
    (r"\boxed{\frac{1}{2}}", None, "1/2", "none", "boxed"),
    # --- strict numbers -------------------------------------------------------------------
    (r"\boxed{0}", "0", "0", "boxed", "boxed"),
    (r"\boxed{007}", "7", "7", "boxed", "boxed"),
    (r"\boxed{-0.25}", "-1/4", "-1/4", "boxed", "boxed"),
    (r"\boxed{1234567}", "1234567", "1234567", "boxed", "boxed"),
    (r"\boxed{3.14159}", "3.14159", "3.14159", "boxed", "boxed"),
    ("Therefore $\\boxed{18}$.", "18", "18", "boxed", "boxed"),
    (r"\boxed {18}", "18", "18", "boxed", "boxed"),
    ("\\boxed{\n18\n}", "18", "18", "boxed", "boxed"),
    (r"\boxed{\boxed{18}}", "18", "18", "boxed", "boxed"),
    ("\\boxed{4} and then it got cut off \\boxed{5", "4", "4", "boxed", "boxed"),
    ("\\boxed{18}\n\n**Final answer: 20**", "18", "18", "boxed", "boxed"),
    ("Natalia sold 48/2 = 24 clips in May, so 48+24 = 72 altogether.\n#### 72", "72", "72", "hash", "hash"),
    # --- lenient box contents -------------------------------------------------------------
    (r"\boxed{1,234,567}", None, "1234567", "none", "boxed"),
    (r"\boxed{-1,000}", None, "-1000", "none", "boxed"),
    (f"\\boxed{{{MINUS}1,000}}", None, "-1000", "none", "boxed"),
    (r"\boxed{\$18}", None, "18", "none", "boxed"),
    (r"\boxed{$18$}", None, "18", "none", "boxed"),
    (r"\boxed{12.5\%}", None, "12.5", "none", "boxed"),
    (r"\boxed{\textbf{18}}", None, "18", "none", "boxed"),
    (r"\boxed{\mathrm{18}}", None, "18", "none", "boxed"),
    (r"\boxed{\text{\textbf{18}}}", None, "18", "none", "boxed"),
    (r"\boxed{18\text{ apples}}", None, "18", "none", "boxed"),
    (r"\boxed{\text{18 apples}}", None, "18", "none", "boxed"),
    (r"\boxed{\mbox{18}}", None, "18", "none", "boxed"),
    (r"\boxed{1{,}234}", None, "1234", "none", "boxed"),
    (r"\boxed{1\,234}", None, "1234", "none", "boxed"),
    (r"\boxed{\$\,1{,}000.00}", None, "1000", "none", "boxed"),
    (r"\boxed{\frac{3}{4}}", None, "3/4", "none", "boxed"),
    (r"\boxed{\tfrac{1}{3}}", None, "1/3", "none", "boxed"),
    (r"\boxed{-\frac{1}{2}}", None, "-1/2", "none", "boxed"),
    (r"\boxed{\frac{1,000}{4}}", None, "250", "none", "boxed"),
    (r"\boxed{3/4}", None, "3/4", "none", "boxed"),
    (r"\boxed{7 / 2}", None, "7/2", "none", "boxed"),
    (r"\boxed{90^\circ}", None, "90", "none", "boxed"),
    (r"\boxed{90^{\circ}}", None, "90", "none", "boxed"),
    (r"\boxed{\left(18\right)}", None, "18", "none", "boxed"),
    (r"\boxed{\{12\}}", None, "12", "none", "boxed"),
    (r"\boxed{x = \frac{5}{2}}", None, "5/2", "none", "boxed"),
    ("\\boxed{\uff14\uff12}", "42", "42", "boxed", "boxed"),  # full-width digits are Unicode \d
    # --- #### lines -----------------------------------------------------------------------
    ("#### 42\n", "42", "42", "hash", "hash"),
    ("#### -7", "-7", "-7", "hash", "hash"),
    ("#### 3.5", "3.5", "3.5", "hash", "hash"),
    ("#### 3.5.", "3.5", "3.5", "hash", "hash"),
    ("#### 42 .", "42", "42", "hash", "hash"),
    ("#### 42..", None, "42", "none", "hash"),
    ("#### 42 #### 43", "43", "43", "hash", "hash"),
    ("##### 42", "42", "42", "hash", "hash"),
    ("#### 42\r\n", "42", "42", "hash", "hash"),
    ("#### 42 dollars.", None, "42", "none", "hash"),
    ("#### $1,234.50", None, "1234.5", "none", "hash"),
    ("#### 1/2.", None, "1/2", "none", "hash"),
    ("Result:\n#### \nWe had 12 apples", None, "12", "none", "last_number"),
    ("#### 18\nThe explanation mentions 99 and 100 afterwards.", "18", "18", "hash", "hash"),
    # --- marker selection and v2 fallback to the OTHER marker -------------------------------
    ("\\boxed{1,234}\n#### 1234", "1234", "1234", "hash", "hash"),
    ("#### 1,234\n\\boxed{1234}", "1234", "1234", "boxed", "boxed"),
    ("\\boxed{18}\n#### eighteen", None, "18", "none", "boxed"),
    ("#### 18\n\\boxed{x}", None, "18", "none", "hash"),
    ("\\boxed{x}\nThe answer is 12", None, "12", "none", "answer_phrase"),
    ("\\boxed{\\text{eighteen}}", None, None, "none", "none"),
    ("\\boxed{12\\}", None, "12", "none", "last_number"),  # escaped brace: unbalanced box is dropped
    # --- \fbox ----------------------------------------------------------------------------
    (r"\fbox{42}", None, "42", "none", "fbox"),
    (r"\fbox{\$1,000}", None, "1000", "none", "fbox"),
    ("\\boxed{} and \\fbox{12}", None, "12", "none", "fbox"),
    # --- answer phrases ---------------------------------------------------------------------
    ("the answer is: 15 apples", None, "15", "none", "answer_phrase"),
    ("The final answer is 2.5.", None, "2.5", "none", "answer_phrase"),
    ("Answer = 64", None, "64", "none", "answer_phrase"),
    ("ANSWER IS 7", None, "7", "none", "answer_phrase"),
    (f"The answer is {MINUS}5", None, "-5", "none", "answer_phrase"),
    ("So the answer is \\frac{1}{4}", None, "1/4", "none", "answer_phrase"),
    ("The answer is 1,234,567.", None, "1234567", "none", "answer_phrase"),
    ("The answer is 8.\nWait, the final answer: 9", None, "9", "none", "answer_phrase"),
    ("**Step 1:** add 2 and 3 to get 5. **Answer:** 17", None, "17", "none", "answer_phrase"),
    # --- bold -------------------------------------------------------------------------------
    ("Total: **$1,200**", None, "1200", "none", "bold"),
    ("We get __36__ cookies", None, "36", "none", "bold"),
    ("**12** apples and **30** pears, **in total**", None, "30", "none", "bold"),
    # --- last number --------------------------------------------------------------------------
    ("Step one yields 12, step two yields 30.", None, "30", "none", "last_number"),
    ("Costs $1,250.75 total", None, "1250.75", "none", "last_number"),
    (f"The temperature drops to {MINUS}12 degrees", None, "-12", "none", "last_number"),
    ("Value .75 is the result", None, "3/4", "none", "last_number"),
    ("Final Answer\n\nThe total is 34", None, "34", "none", "last_number"),
    ("\uff11\uff12 apples", None, "12", "none", "last_number"),
    ("x2 y3", None, None, "none", "none"),
    ("", None, None, "none", "none"),
]


def _frac(s: str | None) -> Fraction | None:
    return None if s is None else Fraction(s)


def _id(case: tuple) -> str:
    return repr(case[0])[:48]


def test_golden_table_size() -> None:
    assert len(GOLDEN) >= 80


@pytest.mark.parametrize(("text", "v1", "v2", "m1", "m2"), GOLDEN, ids=[_id(c) for c in GOLDEN])
def test_golden(text: str, v1: str | None, v2: str | None, m1: str, m2: str) -> None:
    a, b = extract_v1(text), extract_v2(text)
    assert (a.value, a.method) == (_frac(v1), m1)
    assert (b.value, b.method) == (_frac(v2), m2)
    for ex in (a, b):
        assert ex.method in METHODS
        if ex.span is not None:  # spans point at the parsed content of the ORIGINAL text
            assert text[ex.span[0] : ex.span[1]] == ex.content
        if ex.value is not None:
            assert ex.span is not None and ex.content is not None
    if a.value is not None:  # monotone: v2 reads the same marker payload
        assert (b.value, b.method, b.span) == (a.value, a.method, a.span)


@pytest.mark.parametrize(
    ("text", "gold", "v1_ok", "v2_ok"),
    [
        (r"\boxed{5.00}", "5", True, True),
        (r"\boxed{5.00}", 5, True, True),
        (r"\boxed{4.99}", "5", False, False),
        (r"\boxed{1,234}", "1,234", False, True),
        (r"\boxed{1,234}", 1234, False, True),
        (r"\boxed{\frac{1}{3}}", Fraction(1, 3), False, True),
        (r"\boxed{\frac{1}{3}}", "1/3", False, True),
        (r"\boxed{0.3333}", "1/3", False, False),
        ("#### 72", "72", True, True),
        ("no answer here", "0", False, False),
    ],
)
def test_is_correct(text: str, gold: object, v1_ok: bool, v2_ok: bool) -> None:
    assert is_correct(extract_v1(text), gold) is v1_ok
    assert is_correct(extract_v2(text), gold) is v2_ok


def test_v1_reports_rejected_marker_content() -> None:
    ex = extract_v1("so \\boxed{1,234}")
    assert ex == Extraction(None, "none", (10, 15), "1,234")
    assert extract_v1("nothing to see") == NO_EXTRACTION
    assert NO_EXTRACTION.extracted is None


def test_last_number_span_maps_to_original_text() -> None:
    text = "about \uff11\uff12 apples (\ufb01ne)"
    ex = extract_v2(text)
    assert ex.value == 12 and ex.span == (6, 8) and ex.content == "\uff11\uff12"
    # Combining sequences compose under whole-text NFKC: value still found, span unknown.
    ex = extract_v2("cafe\u0301 costs 5")
    assert (ex.value, ex.method, ex.span, ex.content) == (Fraction(5), "last_number", None, "5")


def test_extracted_property_is_canonical() -> None:
    assert extract_v2(r"\boxed{\$1,234.50}").extracted == "1234.5"
    assert extract_v2(r"\boxed{\frac{1}{3}}").extracted == "1/3"
    assert extract_v1(r"\boxed{5.00}").extracted == "5"


# --------------------------------------------------------------------------- shared helpers


def test_find_boxes() -> None:
    assert find_boxes(r"\boxed{1} x \boxed{ 2 }") == [("1", 7, 8), (" 2 ", 19, 22)]
    text = r"\boxed{\frac{1}{2}}"
    assert find_boxes(text) == [(r"\frac{1}{2}", 7, 18)]
    assert find_boxes(r"\boxed{a\}b}") == [(r"a\}b", 7, 11)]  # escaped brace is literal
    assert find_boxes(r"\boxed{a\{b}") == [(r"a\{b", 7, 11)]
    assert find_boxes(r"\boxed{4") == []  # truncated
    assert find_boxes(r"\boxed{a\\}") == [(r"a\\", 7, 10)]  # "\\" is an escaped backslash
    nested = r"\boxed{\boxed{3}}"
    assert [c for c, _, _ in find_boxes(nested)] == [r"\boxed{3}", "3"]
    assert find_boxes("\\boxed\n{7}") == [("7", 8, 9)]
    assert find_boxes("no boxes") == []


_BRACE_BITS = st.sampled_from(["\\boxed{", "\\boxed {", "{", "}", "\\{", "\\}", "\\", "\\\\", "x", "1"])


@given(st.lists(_BRACE_BITS, max_size=30).map("".join))
@settings(derandomize=True, database=None, max_examples=500)
def test_find_boxes_matches_depth_scan(text: str) -> None:
    """The linear brace pairing equals a depth scan started at every box's opening brace."""
    expected = []
    for m in BOX_RE.finditer(text):
        close = match_brace(text, m.end() - 1)
        if close is not None:
            expected.append((text[m.end() : close], m.end(), close))
    assert find_boxes(text) == expected


def test_last_hash_line() -> None:
    text = "#### 1\nmore\n####  42 apples  \nend"
    content, start, end = last_hash_line(text)
    assert content == "42 apples" and text[start:end] == content
    assert last_hash_line("no marker") is None
    assert last_hash_line("####") == ("", 4, 4)


def test_select_marker_prefers_later_marker() -> None:
    assert select_marker(r"\boxed{5} then #### 7") == ("hash", "7", (20, 21))
    assert select_marker(r"#### 7 then \boxed{ 5 }") == ("boxed", "5", (20, 21))
    assert select_marker(r"#### \boxed{5}") == ("boxed", "5", (12, 13))
    assert select_marker(r"\boxed{#### 5}")[0] == "hash"
    assert select_marker(r"\boxed{5") is None
    assert select_marker("plain text 5") is None


@pytest.mark.parametrize(
    ("s", "expected"),
    [
        ("42", "42"),
        ("  -3 ", "-3"),
        ("5.", "5"),
        (".5", "1/2"),
        ("-.5", "-1/2"),
        ("1,234", "1234"),
        ("\\$1,234.50", "1234.5"),
        ("{{{7}}}", "7"),
        ("{1}{2}", "1"),
        ("a = b = 9", "9"),
        ("\\frac{1}{0}", None),
        ("\\frac{6}{4}", "3/2"),
        ("-\\frac{-1}{2}", "1/2"),
        ("10 / 4", "5/2"),
        ("1/0", None),
        ("\\text{\\text{\\text{12}}}", "12"),
        ("twelve", None),
        ("", None),
    ],
)
def test_parse_lenient(s: str, expected: str | None) -> None:
    assert parse_lenient(s) == _frac(expected)


# --------------------------------------------------------------------------- canonical / gold


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (Fraction(18), "18"),
        (Fraction(-3), "-3"),
        (Fraction(0), "0"),
        (Fraction(1, 2), "0.5"),
        (Fraction(-1, 2), "-0.5"),
        (Fraction(2469, 2), "1234.5"),
        (Fraction(1, 8), "0.125"),
        (Fraction(1, 40), "0.025"),
        (Fraction(1, 1024), "0.0009765625"),
        (Fraction(10**20), "100000000000000000000"),
        (Fraction(1, 3), "1/3"),
        (Fraction(-1, 3), "-1/3"),
        (Fraction(-7, 6), "-7/6"),
        (7, "7"),
    ],
)
def test_canonical(value: Fraction, text: str) -> None:
    assert canonical(value) == text


@given(st.fractions(max_denominator=10**6))
@settings(derandomize=True, database=None, max_examples=300)
def test_canonical_roundtrip(value: Fraction) -> None:
    s = canonical(value)
    assert gold_value(s) == value
    assert "e" not in s.lower() and not ("." in s and s.endswith("0"))


@pytest.mark.parametrize(
    ("gold", "expected"),
    [
        ("18", Fraction(18)),
        ("1,234", Fraction(1234)),
        (" 5.00 ", Fraction(5)),
        ("-0.5", Fraction(-1, 2)),
        (7, Fraction(7)),
        (Fraction(1, 3), Fraction(1, 3)),
        ("1/3", Fraction(1, 3)),
        (Decimal("2.50"), Fraction(5, 2)),
    ],
)
def test_gold_value(gold: object, expected: Fraction) -> None:
    assert gold_value(gold) == expected


def test_parse_gold() -> None:
    assert parse_gold("Natalia sold 48/2 = <<48/2=24>>24 clips.\n#### 72") == 72
    assert parse_gold("#### 2,125") == 2125
    assert parse_gold("x\n#### -5") == -5
    assert parse_gold("a #### b\n#### 0.5") == Fraction(1, 2)


def _gold_rows() -> list[dict]:
    rows = []
    for path in sorted((REPO / "data" / "gsm8k").glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return rows


def test_all_400_golds_parse() -> None:
    rows = _gold_rows()
    assert len(rows) == 400
    assert {r["split"] for r in rows} == {"train", "test"}
    for r in rows:
        g = parse_gold(r["answer"])
        assert g == gold_value(r["gold"]), r["idx"]
        assert g.denominator == 1, r["idx"]
        assert canonical(g) == r["gold"], r["idx"]


# --------------------------------------------------------------------------- registry and hashes


def test_registry_and_dispatch() -> None:
    assert list(REGISTRY) == ["v1", "v2"]
    assert REGISTRY["v1"] is extract_v1 and REGISTRY["v2"] is extract_v2
    assert extract("v1", r"\boxed{1,234}").value is None
    assert extract("v2", r"\boxed{1,234}").value == 1234
    with pytest.raises(KeyError, match="unknown extractor"):
        extract("v3", "x")
    with pytest.raises(KeyError):
        extractor_hash("v3")


def test_hash_and_tag_format() -> None:
    for name in REGISTRY:
        h = extractor_hash(name)
        assert len(h) == 12 and all(c in "0123456789abcdef" for c in h)
        assert extractor_tag(name) == f"{name}@{h[:8]}"
    assert extractor_hash("v1") != extractor_hash("v2")
    assert extractor_tags() == {n: extractor_tag(n) for n in REGISTRY}


def test_hash_normalises_line_endings() -> None:
    src = REPO / "src" / "driftlab" / "extraction"
    common, v1 = (src / "common.py").read_bytes(), (src / "v1.py").read_bytes()
    crlf = [b.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n") for b in (common, v1)]
    assert hash_sources(crlf)[:12] == hash_sources([common, v1])[:12] == extractor_hash("v1")


def test_extractors_are_frozen() -> None:
    pinned = json.loads(FROZEN_JSON.read_text(encoding="utf-8"))
    mismatches = frozen_mismatches(pinned)
    assert not mismatches, (
        f"Extractor source hash changed: {mismatches} (name: (pinned, current)). The extractors are a frozen "
        "research artifact: stored scores reference this hash (scores.ext_hash, environment fingerprints). "
        "Do NOT edit common.py / v1.py / v2.py or re-pin this file; add the new behaviour as a NEW version "
        "(e.g. extraction/v3.py registered as 'v3') and pin its hash in tests/extractors_frozen.json."
    )
    assert verify_frozen(FROZEN_JSON) == {}


def test_frozen_mismatches_reports_both_sides() -> None:
    current = {n: extractor_hash(n) for n in REGISTRY}
    assert frozen_mismatches(current) == {}
    assert frozen_mismatches({**current, "v1": "000000000000"}) == {"v1": ("000000000000", current["v1"])}
    assert frozen_mismatches({"v2": current["v2"]}) == {"v1": (None, current["v1"])}
    assert frozen_mismatches({**current, "v9": "abc"}) == {"v9": ("abc", None)}


# --------------------------------------------------------------------------- monotonicity property

_INTS = st.integers(-100_000, 10**7).map(str)
_DECIMALS = st.tuples(st.integers(-9_999, 9_999), st.integers(0, 999)).map(lambda t: f"{t[0]}.{t[1]}")
_COMMAS = st.integers(1_000, 10**8).map(lambda n: f"{n:,}")
_ODD = st.sampled_from([".5", "5.", "-.25", f"{MINUS}3", "\uff14\uff12", "007", "-0", "1/2", "3.0.1"])
_NUMBER = st.one_of(_INTS, _DECIMALS, _COMMAS, _ODD)
_WRAPS = st.sampled_from(
    [
        "{}",
        " {} ",
        "{}.",
        "\\${}",
        "${}$",
        "{}\\%",
        "{} \\text{{ dollars}}",
        "\\text{{{}}}",
        "\\textbf{{{}}}",
        "x = {}",
        "3 \\times 6 = {}",
        "{{{}}}",
        "\\frac{{{}}}{{4}}",
        "{} apples",
        "about {}",
        "{}{{,}}000",
    ]
)
_CONTENT = st.one_of(
    _NUMBER,
    st.tuples(_WRAPS, _NUMBER).map(lambda t: t[0].format(t[1])),
    st.text(alphabet="0123456789 ,.-$x=\\{}%", max_size=8),
)
_FILLER = st.one_of(
    st.text(alphabet=" abcxyz0123456789.,-$#*_{}\\\n=:/%" + MINUS, max_size=24),
    st.sampled_from(
        [
            "####",
            "\\boxed",
            "\\boxed{",
            "{",
            "}",
            "\\{",
            "\\}",
            "\\",
            "**",
            "__",
            "\n",
            "The answer is ",
            ". ",
        ]
    ),
)
_FRAGMENT = st.one_of(
    _FILLER,
    _CONTENT.map(lambda c: f"\\boxed{{{c}}}"),
    _CONTENT.map(lambda c: f"\\boxed{{{c}"),  # truncated
    st.tuples(_CONTENT, st.sampled_from(["", ".", " ", ". ", "\n"])).map(lambda t: f"\n#### {t[0]}{t[1]}"),
    _CONTENT.map(lambda c: f"\\fbox{{{c}}}"),
    _CONTENT.map(lambda c: f"**{c}**"),
    _CONTENT.map(lambda c: f"Final Answer: {c}\n"),
)
RESPONSES = st.lists(_FRAGMENT, max_size=10).map("".join)
_GOLDS = st.one_of(st.integers(-1_000, 10**6), st.fractions(max_denominator=8).map(canonical))


def _assert_monotone(text: str, golds: list) -> None:
    a, b = extract_v1(text), extract_v2(text)
    if a.value is not None:
        assert (b.value, b.method, b.span) == (a.value, a.method, a.span)
        golds = [*golds, a.value]
    for gold in golds:
        assert not is_correct(a, gold) or is_correct(b, gold), (text, gold)


@given(RESPONSES, _GOLDS)
@settings(derandomize=True, database=None, max_examples=2000, suppress_health_check=[HealthCheck.too_slow])
def test_monotone_on_marker_rich_texts(text: str, gold: object) -> None:
    _assert_monotone(text, [gold])


@given(st.text(max_size=200), _GOLDS)
@settings(derandomize=True, database=None, max_examples=500)
def test_monotone_and_total_on_arbitrary_text(text: str, gold: object) -> None:
    _assert_monotone(text, [gold])
    for fn in (extract_v1, extract_v2):
        ex = fn(text)
        assert ex.method in METHODS
        assert (ex.value is None) == (ex.method == "none")


@given(st.from_regex(r"-?\d+(?:\.\d+)?", fullmatch=True))
@settings(derandomize=True, database=None, max_examples=500)
def test_parse_lenient_agrees_with_strict_numbers(s: str) -> None:
    """The lemma behind monotonicity: parse_lenient(s) == Fraction(Decimal(s)) for every v1-valid s."""
    assert parse_lenient(s) == Fraction(Decimal(s))
    assert parse_lenient(s + ".") == Fraction(Decimal(s))
