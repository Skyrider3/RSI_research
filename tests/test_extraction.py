"""Golden, property and frozen-hash tests for the answer extractors (a frozen research artifact)."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections.abc import Iterator
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
    is_hash_answer,
    last_hash_line,
    parse_gold,
    parse_lenient,
    select_marker,
    verify_frozen,
)
from driftlab.extraction import _hash_sources as hash_sources
from driftlab.extraction.common import BOX_RE, HASH_ANSWER_RE, last_number, match_brace, other_marker
from driftlab.extraction.v2 import BOLD_NUMERIC_RE

REPO = Path(__file__).resolve().parents[1]
FROZEN_JSON = Path(__file__).with_name("extractors_frozen.json")
MINUS = "\u2212"

# The project lead's literal patterns; the shipped ones are backtracking-safe rewrites of the same languages.
SPEC_HASH_ANSWER_RE = re.compile(r"[-+\u2212]?\s*(?:\\?\$)?\s*[-\u2212]?\s*(?:\d|\.\d)")
SPEC_NUMBERED_HEADING_RE = re.compile(r"\d+[.)]\s+[A-Za-z]")
SPEC_BOLD_NUMERIC_RE = re.compile(
    r"^\s*(?:[A-Za-z]\s*=\s*)?[-+\u2212]?\s*(?:\\?\$)?\s*[-\u2212]?\s*(?:\d[\d,]*(?:\.\d+)?|\.\d+)\s*"
    r"(?:\\?%|[A-Za-z]+(?:\s+[A-Za-z]+)?)?\s*[.!]?\s*$"
)


def _spec_is_hash_answer(content: str) -> bool:
    """The lead's rule, literally: stripped content starts like a number and is not a numbered heading."""
    c = content.strip()
    return SPEC_HASH_ANSWER_RE.match(c) is not None and SPEC_NUMBERED_HEADING_RE.match(c) is None


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
    ("#### -3", "-3", "-3", "hash", "hash"),
    ("#### .5", None, "1/2", "none", "hash"),
    ("####42", "42", "42", "hash", "hash"),
    ("#### \uff14\uff12", "42", "42", "hash", "hash"),  # full-width digits are Unicode \d
    ("#### +5", None, "5", "none", "hash"),
    (f"#### {MINUS}3", None, "-3", "none", "hash"),
    ("#### \\$42", None, "42", "none", "hash"),
    ("#### -$5", None, "-5", "none", "hash"),
    ("#### $-5", None, "-5", "none", "hash"),
    ("#### $ 1,000.", None, "1000", "none", "hash"),
    ("#### 4.", "4", "4", "hash", "hash"),  # a number with a final '.' is not a numbered heading
    ("#### -.5", None, "-1/2", "none", "hash"),
    ("#### $.5", None, "1/2", "none", "hash"),
    # --- a sign separated from the number by whitespace is kept (parse_lenient) -----------------
    ("#### - 7", None, "-7", "none", "hash"),
    (f"#### {MINUS}  7", None, "-7", "none", "hash"),
    ("#### $ - 7", None, "-7", "none", "hash"),
    ("#### - $7", None, "-7", "none", "hash"),
    ("#### - 1,234.", None, "-1234", "none", "hash"),
    ("**- 7**", None, "-7", "none", "bold"),
    ("So **x = - 7**", None, "-7", "none", "bold"),
    (r"\boxed{- 7}", None, "-7", "none", "boxed"),
    (f"\\boxed{{{MINUS}  7}}", None, "-7", "none", "boxed"),
    (r"\boxed{\$ - 7}", None, "-7", "none", "boxed"),
    (r"\boxed{- .5}", None, "-1/2", "none", "boxed"),
    (r"\boxed{- 7/2}", None, "-7/2", "none", "boxed"),
    (r"\boxed{x = - 7}", None, "-7", "none", "boxed"),
    (r"\boxed{\text{- 7 apples}}", None, "-7", "none", "boxed"),
    (r"\boxed{12 - 7 = 5}", None, "5", "none", "boxed"),  # only a sign at the START is joined
    (r"\boxed{3 - 4}", None, "3", "none", "boxed"),
    ("The answer is - 7", None, "-7", "none", "answer_phrase"),
    (r"\fbox{- 7}", None, "-7", "none", "fbox"),
    # --- only QUALIFYING "####" lines are markers (rest of the line starts like a number) ------
    ("\\boxed{72}\n\n#### Step 4: Verify", "72", "72", "boxed", "boxed"),
    ("\\boxed{72}\n\n#### Step 4: Verify\nWe check it.", "72", "72", "boxed", "boxed"),
    ("\\boxed{72}\n\n#### Verification\nWe check 72 / 2 = 36.", "72", "72", "boxed", "boxed"),
    ("\\boxed{18}\n#### Final Answer", "18", "18", "boxed", "boxed"),
    ("\\boxed{18}\n####\n", "18", "18", "boxed", "boxed"),
    ("\\boxed{5} ... #### 7", "7", "7", "hash", "hash"),
    ("#### 42\n\n#### Verification\nWe check 42 / 2 = 21.", "42", "42", "hash", "hash"),
    ("#### 42\n####", "42", "42", "hash", "hash"),
    ("#### 42\n#### Step 5: Check\n\\boxed{x}", None, "42", "none", "hash"),
    ("#### \\boxed{5}", "5", "5", "boxed", "boxed"),
    ("#### Step 3", None, "3", "none", "last_number"),
    ("#### Step 3\nAdd 4 and 5 to get 9.", None, "9", "none", "last_number"),
    ("#### Step 3\nSo the answer is 12.", None, "12", "none", "answer_phrase"),
    ("#### Step 1: Cost\nIt is 5.\n#### Step 2: Multiply\n5 x 3 = 15 so", None, "15", "none", "last_number"),
    ("#### Final Answer\n42", None, "42", "none", "last_number"),
    ("#### Final Answer: 42", None, "42", "none", "answer_phrase"),
    ("#### Final Answer\n**42**", None, "42", "none", "bold"),
    ("#### The answer is 42", None, "42", "none", "answer_phrase"),
    ("#### **42**", None, "42", "none", "bold"),
    ("#### x = 5", None, "5", "none", "last_number"),
    ("#### Verification", None, None, "none", "none"),
    ("####", None, None, "none", "none"),
    ("### Step 1\n#### Step 2\n\\boxed{6}", "6", "6", "boxed", "boxed"),
    # numberless lines: a '.' only qualifies when a digit follows it ("#### .5" does)
    ("\\boxed{18}\n#### ...", "18", "18", "boxed", "boxed"),
    ("\\boxed{18}\n#### .", "18", "18", "boxed", "boxed"),
    ("\\boxed{18}\n#### -.", "18", "18", "boxed", "boxed"),
    ("\\boxed{18}\n#### $.", "18", "18", "boxed", "boxed"),
    ("\\boxed{18}\n#### .5", None, "1/2", "none", "hash"),
    ("#### ...", None, None, "none", "none"),
    ("#### ...\nSo we get 42", None, "42", "none", "last_number"),
    ("#### 42\n#### ...", "42", "42", "hash", "hash"),
    # numbered headings ("4. Verification", "2) Check") are not markers
    ("\\boxed{72}\n\n#### 4. Verification\nWe check 72 / 2 = 36.", "72", "72", "boxed", "boxed"),
    ("\\boxed{72}\n\n#### 2) Check the answer", "72", "72", "boxed", "boxed"),
    ("#### 1. Understand the problem\nWe need 5 + 7 = 12", None, "12", "none", "last_number"),
    ("#### 42\n\n#### 4. Verification\nWe check 42 / 2 = 21.", "42", "42", "hash", "hash"),
    ("#### 42.\n#### 5) Done", "42", "42", "hash", "hash"),
    ("#### 10. Final answer: 42", None, "42", "none", "answer_phrase"),
    ("#### 3. Check\n**18**", None, "18", "none", "bold"),
    ("#### 2.\tVerify", None, "2", "none", "last_number"),  # any whitespace after the '.'
    ("#### 9 #### 4) a", None, "9", "none", "hash"),  # last "####" is a heading; the earlier one is not
    ("#### 3.5 hours", None, "3.5", "none", "hash"),  # a decimal, not a heading
    # --- marker selection and v2 fallback to the OTHER marker -------------------------------
    ("\\boxed{1,234}\n#### 1234", "1234", "1234", "hash", "hash"),
    ("#### 1,234\n\\boxed{1234}", "1234", "1234", "boxed", "boxed"),
    ("\\boxed{18}\n#### eighteen", "18", "18", "boxed", "boxed"),  # "eighteen" does not qualify
    ("\\boxed{18}\n#### 7 = x", None, "18", "none", "boxed"),  # qualifies; parse reads "x": v2 -> box
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
    # --- bold (only number-like spans) --------------------------------------------------------
    ("Total: **$1,200**", None, "1200", "none", "bold"),
    ("We get __36__ cookies", None, "36", "none", "bold"),
    ("**12** apples and **30** pears, **in total**", None, "30", "none", "bold"),
    ("Total: **$1,234**", None, "1234", "none", "bold"),
    ("The cost is **18 dollars**", None, "18", "none", "bold"),
    ("**25%**", None, "25", "none", "bold"),
    ("So **x = 18**", None, "18", "none", "bold"),
    ("It takes **3.5 hours.**", None, "3.5", "none", "bold"),
    ("**-7**", None, "-7", "none", "bold"),
    ("**\\$5**", None, "5", "none", "bold"),
    ("**12 red apples**", None, "12", "none", "bold"),
    ("__1,000__ meters", None, "1000", "none", "bold"),
    ("**18**\n\n**Step 4: Verify the answer**", None, "18", "none", "bold"),
    ("**Step 3: Add the parts**\n5 + 13 = 18 and then we", None, "18", "none", "last_number"),
    ("**Total = 5 + 13 = 18**", None, "18", "none", "last_number"),
    ("**Total = 5 + 13 = 18** so we buy 2 more", None, "2", "none", "last_number"),
    ("**12 big red apples**", None, "12", "none", "last_number"),  # at most two unit words
    # --- last number --------------------------------------------------------------------------
    ("Step one yields 12, step two yields 30.", None, "30", "none", "last_number"),
    ("Costs $1,250.75 total", None, "1250.75", "none", "last_number"),
    (f"The temperature drops to {MINUS}12 degrees", None, "-12", "none", "last_number"),
    ("Value .75 is the result", None, "3/4", "none", "last_number"),
    ("Final Answer\n\nThe total is 34", None, "34", "none", "last_number"),
    ("\uff11\uff12 apples", None, "12", "none", "last_number"),
    ("x2 y3", None, None, "none", "none"),
    ("", None, None, "none", "none"),
    # --- realistic Qwen-style endings -----------------------------------------------------
    ("Therefore, the answer is \\(\\boxed{72}\\).", "72", "72", "boxed", "boxed"),
    ("\\[\n\\boxed{72}\n\\]", "72", "72", "boxed", "boxed"),
    ("### Final Answer\n\\boxed{72}", "72", "72", "boxed", "boxed"),
    ("#### Final Answer\nThe total is \\(\\boxed{72}\\).", "72", "72", "boxed", "boxed"),
    # --- documented quirks (frozen; a fix needs a NEW version) -------------------------------
    # NUM_RE has no lookahead: a run of comma groups is one number.
    ("We have 120,150,180 in total", None, "120150180", "none", "last_number"),
    # The LAST box is selected even when it is empty.
    ("\\boxed{18}\nThe final answer is \\boxed{}", None, "18", "none", "last_number"),
    # The numbered-heading rule needs whitespace after the '.' / ')' (lead's pattern), so a run-together
    # heading qualifies and shadows an earlier box; a heading whose text starts with a digit qualifies too.
    ("\\boxed{72}\n#### 4.Verification", None, "4", "none", "hash"),
    ("\\boxed{72}\n#### 4. 5 apples", None, "4", "none", "hash"),
    # The spaced sign is joined only when a NUMBER follows it, not a \frac (the frac's numerator is read).
    (r"\boxed{- \frac{1}{2}}", None, "1", "none", "boxed"),
    # The bold stage takes the LAST NUMBER-LIKE span: a later non-number-like span is skipped, not a stop.
    ("We buy **5** apples. Total: **Total: 18**", None, "5", "none", "bold"),
    # A fraction is not number-like bold content; the last-number stage reads the denominator.
    ("**1/2**", None, "2", "none", "last_number"),
]


def _frac(s: str | None) -> Fraction | None:
    return None if s is None else Fraction(s)


def _id(case: tuple) -> str:
    return repr(case[0])[:48]


def test_golden_table_size() -> None:
    assert len(GOLDEN) >= 80


def test_golden_table_covers_every_method() -> None:
    """Every v2 stage (and both v1 outcomes per marker kind) is pinned by at least one golden case."""
    assert {c[4] for c in GOLDEN} == set(METHODS)
    assert {c[3] for c in GOLDEN} == {"boxed", "hash", "none"}


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
    assert extract_v1("#### $42") == Extraction(None, "none", (5, 8), "$42")
    assert extract_v1("#### Step 4: Verify") == NO_EXTRACTION  # not a marker at all
    assert extract_v1("#### 4. Verification") == NO_EXTRACTION
    assert extract_v1("#### ...") == NO_EXTRACTION
    assert extract_v1("#### - 7") == Extraction(None, "none", (5, 8), "- 7")
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


def test_huge_numbers_do_not_hit_the_int_str_limit() -> None:
    """Regression: canonical() used str(int), which raises ValueError above 4300 digits (Python >= 3.10.7),
    so ``Extraction.extracted`` (what scoring stores) crashed on a long digit run."""
    big = "9" * 5000
    for text in (f"\\boxed{{{big}}}", f"#### {big}", f"so {big}", f"\\boxed{{0.{big}}}"):
        for fn in (extract_v1, extract_v2):
            ex = fn(text)
            if ex.value is not None:
                assert gold_value(ex.extracted) == ex.value
    assert extract_v1(f"\\boxed{{{big}}}").extracted == big
    assert extract_v2(f"\\boxed{{-{big}.50}}").extracted == f"-{big}.5"
    assert extract_v2(f"\\boxed{{\\frac{{1}}{{3{'0' * 5000}}}}}").extracted == f"1/3{'0' * 5000}"
    assert canonical(Fraction(1, 2**5000)) == "0." + str(5**5000).rjust(5000, "0")  # 3495 digits
    assert canonical(Fraction(-(10**6000) - 1, 3)) == "-1" + "0" * 5999 + "1/3"
    assert gold_value(canonical(Fraction(7, 10**6000))) == Fraction(7, 10**6000)


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
    assert last_hash_line("####") is None  # empty content never qualifies
    assert last_hash_line("#### \n") is None
    assert last_hash_line("#### 7\n#### Step 4: Verify\n####\n#### Final Answer\n") == ("7", 5, 6)
    assert last_hash_line("#### .5") == (".5", 5, 7)
    assert last_hash_line("#### ...") is None and last_hash_line("#### .\n#### -.") is None
    assert last_hash_line("#### 4. Verification") is None and last_hash_line("#### 2) Check") is None
    assert last_hash_line("#### 4.") == ("4.", 5, 7)
    assert last_hash_line("#### 42\n#### 4. Verification\n#### ...") == ("42", 5, 7)
    assert last_hash_line("##### 42") == ("42", 6, 8)  # overlapping occurrence: last one reads " 42"
    assert last_hash_line("#### 1 #### Step 2") == ("1 #### Step 2", 5, 18)  # rest of the line
    assert last_hash_line("#### 1 #### 2. Check") == ("1 #### 2. Check", 5, 20)
    assert last_hash_line("#### $\n5") is None  # the qualification test never crosses the newline
    assert last_hash_line("#### .\n5") is None and last_hash_line("#### 4.\nA") == ("4.", 5, 7)


@pytest.mark.parametrize(
    ("content", "ok"),
    [
        ("42", True),
        ("  -3 ", True),
        (".5", True),
        ("$42", True),
        ("\\$42", True),
        ("$ 1,234", True),
        ("-$5", True),
        ("$-5", True),
        ("+ $ - 5", True),
        (f"{MINUS}3", True),
        ("42 apples", True),
        ("1,234.", True),
        ("\uff14\uff12", True),
        ("-.5", True),
        ("$ .5", True),
        ("4.", True),  # numbers with a final '.' and decimals are not numbered headings
        ("42.", True),
        ("3.5", True),
        ("3.5 hours", True),
        ("4) ", True),
        ("4. 5 apples", True),  # documented: the heading text must start with a letter
        ("4.Verification", True),  # documented: whitespace after the '.' is required
        ("- 7", True),
        ("$ - 7", True),
        ("...", False),  # a '.' qualifies only when a digit follows
        (".", False),
        ("-.", False),
        ("$.", False),
        ("- .", False),
        (". 5", False),
        ("4. Verification", False),  # numbered headings
        ("2) Check the answer", False),
        ("10) Final answer: 42", False),
        ("4.\t Check", False),
        ("1. a", False),
        ("\uff14. Verification", False),  # full-width digits are Unicode \d
        ("# 5", False),  # "#####" heading: the earlier, overlapping occurrence reads "# 5"
        ("Step 4: Verify the 3 parts", False),
        ("", False),
        ("   ", False),
        ("-", False),
        ("$", False),
        ("Step 4: Verify", False),
        ("Final Answer", False),
        ("Verification", False),
        ("**42**", False),
        ("x = 5", False),
        ("\\boxed{5}", False),
        ("#### 5", False),
        ("--5", True),
        ("---5", False),
    ],
)
def test_is_hash_answer(content: str, ok: bool) -> None:
    assert is_hash_answer(content) is ok
    assert _spec_is_hash_answer(content) is ok
    assert is_hash_answer(f" \t{content}\u2003 ") is ok  # surrounding blanks never matter


@pytest.mark.parametrize(
    ("content", "ok"),
    [
        ("42", True),
        ("$1,234", True),
        ("18 dollars", True),
        ("25%", True),
        ("25\\%", True),
        ("x = 18", True),
        ("3.5 hours.", True),
        ("-7", True),
        ("  42  ", True),
        (".5", True),
        ("\\$5", True),
        ("12 red apples", True),
        ("42!", True),
        (f"{MINUS}7", True),
        ("Step 3: Add the parts", False),
        ("Total = 5 + 13 = 18", False),
        ("12 big red apples", False),
        ("Final answer: 20", False),
        ("1/2", False),
        ("in total", False),
        ("", False),
        ("x = y = 3", False),
        ("42 is the answer", False),
    ],
)
def test_bold_numeric_re(content: str, ok: bool) -> None:
    assert (BOLD_NUMERIC_RE.match(content) is not None) is ok
    assert (SPEC_BOLD_NUMERIC_RE.match(content) is not None) is ok


def test_bold_span_is_stripped_content() -> None:
    text = "Total: ** 42 **"
    ex = extract_v2(text)
    assert (ex.value, ex.method, ex.content) == (Fraction(42), "bold", "42")
    assert ex.span == (text.index("42"), text.index("42") + 2)


def _naive_last_hash(text: str) -> tuple[int, str, tuple[int, int]] | None:
    """Reference: every (overlapping) "####" occurrence, rest of its line stripped, spec regex; keep the last.
    Returns (marker start, stripped content, content span)."""
    found = None
    for pos in range(len(text) - 3):
        if text.startswith("####", pos):
            start = pos + 4
            nl = text.find("\n", start)
            raw = text[start : len(text) if nl < 0 else nl]
            content = raw.strip()
            if _spec_is_hash_answer(content):
                lead = len(raw) - len(raw.lstrip())
                found = (pos, content, (start + lead, start + lead + len(content)))
    return found


def _naive_last_hash_line(text: str) -> tuple[str, int, int] | None:
    found = _naive_last_hash(text)
    return None if found is None else (found[1], *found[2])


def _naive_last_box(text: str) -> tuple[int, str, tuple[int, int]] | None:
    """Reference: depth scan from every ``\\boxed{``; the last balanced one, content stripped."""
    found = None
    for m in BOX_RE.finditer(text):
        close = match_brace(text, m.end() - 1)
        if close is not None:
            raw = text[m.end() : close]
            lead = len(raw) - len(raw.lstrip())
            found = (m.start(), raw.strip(), (m.end() + lead, m.end() + lead + len(raw.strip())))
    return found


_HASHY = st.one_of(
    st.text(alphabet="# \t\n\r\u00a0-+$\\.5a)" + MINUS, max_size=40),
    st.lists(
        st.sampled_from(
            [
                "####",
                "#### ",
                "#####",
                "#### Step 3",
                "#### Final Answer",
                "#### 4. Verification",
                "#### 2) Check",
                "#### 10.\tDone",
                "#### 4.",
                "#### 3.5",
                "#### .5",
                "#### ...",
                "#### -.",
                "#### - 7",
                "### Step 2",
                "4. A",
                "7)",
                " a",
                "\n",
                " ",
                "-",
                "$",
                "\\$",
                ".",
                "..",
                "42",
                "1,234",
                "x",
                "\u2003",
                "\r\n",
                "\\boxed{7}",
            ]
        ),
        max_size=14,
    ).map("".join),
)


@given(_HASHY)
@settings(derandomize=True, database=None, max_examples=1500)
def test_last_hash_line_matches_naive_reference(text: str) -> None:
    assert last_hash_line(text) == _naive_last_hash_line(text)
    marker = select_marker(text)
    if marker is not None and marker[0] == "hash":
        assert is_hash_answer(marker[1]) and text[marker[2][0] : marker[2][1]] == marker[1]


_MARKER_MIX = st.lists(
    st.sampled_from(
        [
            "\\boxed{7}",
            "\\boxed{ 1,234 }",
            "\\boxed{x}",
            "\\boxed{",
            "\\boxed {",
            "}",
            "{",
            "\\}",
            "#### 5",
            "#### $42.",
            "#### Step 4: Verify",
            "#### Final Answer",
            "#### 4. Verification",
            "#### 2) Check",
            "#### 4.",
            "#### .5",
            "#### ...",
            "#### -.",
            "#### - 7",
            "#### 7 = x",
            "####",
            "#",
            "\n",
            " ",
            "-",
            "3",
        ]
    ),
    max_size=12,
).map("".join)


@given(_MARKER_MIX)
@settings(derandomize=True, database=None, max_examples=1500)
def test_select_and_other_marker_match_naive_reference(text: str) -> None:
    """Spec oracle: the later of (last balanced box, last QUALIFYING '####'); other_marker = the other kind."""
    box, hsh = _naive_last_box(text), _naive_last_hash(text)
    if box is not None and (hsh is None or box[0] > hsh[0]):
        expected = ("boxed", box[1], box[2])
    elif hsh is not None:
        expected = ("hash", hsh[1], hsh[2])
    else:
        expected = None
    assert select_marker(text) == expected
    assert other_marker(text, "hash") == (None if box is None else ("boxed", box[1], box[2]))
    assert other_marker(text, "boxed") == (None if hsh is None else ("hash", hsh[1], hsh[2]))


# Bold spans exactly as v2 defines them (copied, not imported, so the oracle is independent).
_SPEC_BOLD_SPANS = (re.compile(r"\*\*([^*\n]{1,40})\*\*"), re.compile(r"__([^_\n]{1,40})__"))
_BOLD_BITS = st.one_of(
    st.sampled_from(
        [
            "**42**",
            "**$1,234**",
            "**18 dollars**",
            "**25%**",
            "**x = 18**",
            "**3.5 hours.**",
            "**-7**",
            "** 42 **",
            "**Step 3: Add the parts**",
            "**Total = 5 + 13 = 18**",
            "**12 big red apples**",
            "**in total**",
            "**1/2**",
            "__36__",
            "__init__",
            "**",
            "__",
            "*",
            "_",
            " ",
            "\n",
            "9",
            ", ",
            "apples",
        ]
    ),
    st.integers(0, 10**6).map(lambda n: f"**{n:,}**"),
    st.tuples(st.integers(0, 999), st.sampled_from(["", " cm", " red apples", "%", ".", " = 4"])).map(
        lambda t: f"**{t[0]}{t[1]}**"
    ),
)


def _naive_bold_or_last_number(text: str) -> tuple[Fraction | None, str, str | None]:
    """Spec oracle for texts without markers / fbox / answer phrases: the LAST bold span whose stripped
    content matches the lead's literal pattern and parses; else the last number."""
    spans = sorted((m.span(1) for rx in _SPEC_BOLD_SPANS for m in rx.finditer(text)), reverse=True)
    for s, e in spans:
        content = text[s:e].strip()
        if SPEC_BOLD_NUMERIC_RE.match(content) and (value := parse_lenient(content)) is not None:
            return value, "bold", content
    found = last_number(text)
    return (None, "none", None) if found is None else (found[0], "last_number", found[2])


@given(st.lists(_BOLD_BITS, max_size=10).map("".join))
@settings(derandomize=True, database=None, max_examples=1500)
def test_bold_stage_matches_naive_reference(text: str) -> None:
    ex = extract_v2(text)
    assert (ex.value, ex.method, ex.content) == _naive_bold_or_last_number(text), text


def _all_strings(alphabet: str, max_len: int) -> Iterator[str]:
    """Every string over ``alphabet`` of length 0..max_len (shortest first)."""
    yield ""
    frontier = [""]
    for _ in range(max_len):
        frontier = [s + c for s in frontier for c in alphabet]
        yield from frontier


def test_hash_answer_re_equals_spec_pattern_exhaustively() -> None:
    """Every string of length <= 5 over the pattern's token classes: same .match verdict as the spec."""
    n = 0
    for s in _all_strings(f"-+{MINUS} \\$7.a", 5):
        n += 1
        assert (HASH_ANSWER_RE.match(s) is None) == (SPEC_HASH_ANSWER_RE.match(s) is None), repr(s)
    assert n == sum(9**k for k in range(6))  # 66430 strings


def test_is_hash_answer_equals_spec_rule_exhaustively() -> None:
    """The full qualification rule (number-like start, not a numbered heading) on every string of length
    <= 5 over the heading/number token classes: same verdict as the lead's literal rule, and the linear
    scan's per-line test agrees with it on a "####" line built from the same string."""
    n = 0
    for s in _all_strings(f"7.) a-$\u00a0{MINUS}", 5):
        n += 1
        ok = _spec_is_hash_answer(s)
        assert is_hash_answer(s) is ok, repr(s)
        assert (last_hash_line(f"x\n####{s}\ny") is not None) is ok, repr(s)
    assert n == sum(9**k for k in range(6))  # 66430 strings


def test_bold_numeric_re_equals_spec_pattern_exhaustively() -> None:
    """Every string of length <= 4 over the pattern's token classes: same verdict as the spec."""
    for s in _all_strings(f"x= -{MINUS}+\\$1,.%!", 4):
        assert (BOLD_NUMERIC_RE.match(s) is None) == (SPEC_BOLD_NUMERIC_RE.match(s) is None), repr(s)


@given(st.text(alphabet=f"xab= \t-{MINUS}+\\$15,.%!)", max_size=16))
@settings(derandomize=True, database=None, max_examples=3000)
def test_rewritten_patterns_equal_spec_patterns(s: str) -> None:
    assert (HASH_ANSWER_RE.match(s) is None) == (SPEC_HASH_ANSWER_RE.match(s) is None)
    assert (BOLD_NUMERIC_RE.match(s) is None) == (SPEC_BOLD_NUMERIC_RE.match(s) is None)
    assert is_hash_answer(s) is _spec_is_hash_answer(s)


@pytest.mark.parametrize(
    "text",
    [
        "#" * 60_000,
        "#### Step 3 " * 6_000,
        "#### -" + " " * 6_000 + "x",
        "#### $" + "\t" * 6_000 + "-" + " " * 6_000 + "y\n" * 3,
        ("####" + " " * 50 + "\n") * 1_000,
        ("**" + " " * 19 + "5" + " " * 18 + "x**") * 1_000,
        "\\boxed{1}" + "\n#### Final Answer" * 5_000,
        "\\boxed{1}" + "\n#### 4. Verification" * 5_000,
        "#### 9) a" * 10_000,
        "####" + "9" * 30_000 + ")" + " " * 30_000 + "7",
        ("#### " + "." * 50 + "\n") * 1_000,
        "\\boxed{-" + " \t" * 30_000 + "7}",
        "\\boxed{-" + " " * 60_000 + "x}",
    ],
    ids=[
        "hashes",
        "headings",
        "blank-run",
        "mixed-blank-run",
        "blank-lines",
        "bold-blanks",
        "box-headings",
        "box-numbered-headings",
        "same-line-numbered-headings",
        "heading-backtrack",
        "dot-lines",
        "spaced-sign",
        "spaced-sign-no-number",
    ],
)
def test_pathological_inputs_are_fast(text: str) -> None:
    """Linear marker scan and backtracking-safe patterns: no quadratic/cubic blow-up on long runs."""
    t0 = time.perf_counter()
    for fn in (extract_v1, extract_v2):
        _assert_well_formed(text, fn(text))
    assert time.perf_counter() - t0 < 5.0


def test_select_marker_prefers_later_marker() -> None:
    assert select_marker(r"\boxed{5} then #### 7") == ("hash", "7", (20, 21))
    assert select_marker(r"#### 7 then \boxed{ 5 }") == ("boxed", "5", (20, 21))
    assert select_marker(r"#### \boxed{5}") == ("boxed", "5", (12, 13))
    assert select_marker(r"\boxed{#### 5}")[0] == "hash"
    assert select_marker(r"\boxed{5") is None
    assert select_marker("plain text 5") is None


def test_select_marker_skips_non_qualifying_hash_lines() -> None:
    assert select_marker("\\boxed{72}\n\n#### Step 4: Verify") == ("boxed", "72", (7, 9))
    assert select_marker("\\boxed{72}\n#### Final Answer\n####\n#### Verification") == ("boxed", "72", (7, 9))
    assert select_marker("#### 7\n#### Step 2") == ("hash", "7", (5, 6))
    assert select_marker("#### Final Answer") is None
    assert select_marker("####\n#### \n") is None
    tail = "\n#### 4. Verification\n#### 2) Check\n#### ..."
    assert select_marker("\\boxed{72}" + tail) == ("boxed", "72", (7, 9))
    assert select_marker("#### 4.\n#### 5. Done") == ("hash", "4.", (5, 7))
    assert other_marker("#### 5\n\\boxed{6}\n#### 3) Check", "boxed") == ("hash", "5", (5, 6))


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
        # a leading sign separated from the first number by whitespace is kept
        ("- 7", "-7"),
        (f"{MINUS}  7", "-7"),
        ("-\t\n7", "-7"),
        ("-\u00a07", "-7"),
        ("$ - 7", "-7"),
        ("\\$ - 7", "-7"),
        ("- $ 7", "-7"),
        ("{- 7}", "-7"),
        ("\\text{- 7}", "-7"),
        ("x = - 7", "-7"),
        ("- .5", "-1/2"),
        ("- 1,234.5", "-2469/2"),
        ("- 7/2", "-7/2"),
        ("- 7 - 3", "-7"),
        ("- \uff17", "-7"),
        ("12 - 7 = 5", "5"),  # not at the start: the last '=' decides
        ("3 - 4", "3"),
        ("- 12 - 7 = 5", "5"),
        ("- - 7", "7"),  # the sign must directly precede the number
        ("- x 7", "7"),
        ("- \\frac{1}{2}", "1"),  # documented: a \frac is not a number, so the sign is not joined
        ("-", None),
        ("- ", None),
    ],
)
def test_parse_lenient(s: str, expected: str | None) -> None:
    assert parse_lenient(s) == _frac(expected)


@given(
    st.from_regex(r"[0-9]{1,12}(?:\.[0-9]{1,6})?", fullmatch=True),
    st.sampled_from(["-", MINUS]),
    st.text(alphabet=" \t\n\u00a0\u2003", min_size=1, max_size=6),
    st.sampled_from(
        [("", ""), ("$", ""), ("\\$", ""), ("x = ", ""), ("{", "}"), ("\\text{", "}"), ("", " apples")]
    ),
    st.integers(0, 999),
)
@settings(derandomize=True, database=None, max_examples=500)
def test_parse_lenient_keeps_spaced_leading_sign(
    body: str, sign: str, blanks: str, wrap: tuple[str, str], other: int
) -> None:
    value = Fraction(Decimal(body))
    pre, post = wrap
    assert parse_lenient(f"{pre}{sign}{blanks}{body}{post}") == -value
    assert parse_lenient(f"{sign}{blanks}\\${body}") == -value
    assert parse_lenient(f"{body} {sign} {other}") == value  # a sign between numbers is not joined
    assert parse_lenient(f"{body} {sign} {other} = {other}") == other
    ex = extract_v2(f"#### {sign}{blanks.replace(chr(10), ' ')}{body}")
    assert (ex.value, ex.method) == (-value, "hash")


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


def test_hash_is_sha256_of_common_plus_version_module() -> None:
    """Independent recomputation of the documented definition (not via the private helper)."""
    src = REPO / "src" / "driftlab" / "extraction"
    common = (src / "common.py").read_bytes().replace(b"\r\n", b"\n")
    for name in REGISTRY:
        module = (src / f"{name}.py").read_bytes().replace(b"\r\n", b"\n")
        assert extractor_hash(name) == hashlib.sha256(common + module).hexdigest()[:12]


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
        "research artifact (semantics finalised before the first real run): stored scores reference this hash "
        "(scores.ext_hash, environment fingerprints). Do NOT edit common.py / v1.py / v2.py or re-pin this "
        "file; add the new behaviour as a NEW version (e.g. extraction/v3.py registered as 'v3') and pin its "
        "hash in tests/extractors_frozen.json."
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
_ODD = st.sampled_from(
    [
        ".5",
        "5.",
        "-.25",
        f"{MINUS}3",
        "\uff14\uff12",
        "007",
        "-0",
        "1/2",
        "3.0.1",
        "- 7",
        f"{MINUS}  7",
        "- .5",
    ]
)
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
        "- {}",
        f"{MINUS}\t{{}}",
        "\\$ - {}",
        "- \\${}",
        "x = - {}",
        "12 - {}",
        "{} - 7 = 5",
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
# Markdown headings, numbered headings and bold step headers (never markers / never number-like bold),
# plus numberless '####' lines ("#### ...", "#### -.", which do not qualify either).
_HEADING = st.one_of(
    st.integers(0, 12).map(lambda k: f"\n#### Step {k}: Verify\n"),
    st.integers(0, 12).map(lambda k: f"\n#### Step {k}"),
    st.integers(0, 12).map(lambda k: f"\n#### {k}. Verification\n"),
    st.integers(0, 12).map(lambda k: f"\n#### {k}) Check the answer"),
    st.integers(0, 12).map(lambda k: f"#### {k}. Final answer: "),
    st.integers(0, 12).map(lambda k: f"\n### Step {k}\n"),
    st.integers(0, 12).map(lambda k: f"**Step {k}: Add the parts**"),
    st.integers(0, 12).map(lambda k: f"**Step {k}:** "),
    st.tuples(st.integers(0, 99), st.integers(0, 99)).map(
        lambda t: f"**Total = {t[0]} + {t[1]} = {sum(t)}**"
    ),
    st.sampled_from(
        [
            "\n#### Final Answer\n",
            "\n#### Final Answer",
            "\n#### Verification\n",
            "\n####\n",
            "\n#### \n",
            "#### Final Answer: ",
            "\n#### ...\n",
            "\n#### -.\n",
            "\n#### .",
        ]
    ),
)
_FRAGMENT = st.one_of(
    _FILLER,
    _HEADING,
    _CONTENT.map(lambda c: f"\\boxed{{{c}}}"),
    _CONTENT.map(lambda c: f"\\boxed{{{c}"),  # truncated
    st.tuples(_CONTENT, st.sampled_from(["", ".", " ", ". ", "\n"])).map(lambda t: f"\n#### {t[0]}{t[1]}"),
    _CONTENT.map(lambda c: f"\\fbox{{{c}}}"),
    _CONTENT.map(lambda c: f"**{c}**"),
    _CONTENT.map(lambda c: f"Final Answer: {c}\n"),
)
RESPONSES = st.lists(_FRAGMENT, max_size=10).map("".join)
_GOLDS = st.one_of(st.integers(-1_000, 10**6), st.fractions(max_denominator=8).map(canonical))


def _assert_well_formed(text: str, ex: Extraction) -> None:
    """Contract of every result: known method, value iff method != "none", span indexes the original text,
    and the stored canonical string round-trips to the value."""
    assert ex.method in METHODS
    assert (ex.value is None) == (ex.method == "none"), (text, ex)
    if ex.span is not None:
        assert 0 <= ex.span[0] <= ex.span[1] <= len(text)
        assert text[ex.span[0] : ex.span[1]] == ex.content, (text, ex)
    if ex.value is not None:
        assert ex.content is not None
        assert gold_value(ex.extracted) == ex.value


def _assert_monotone(text: str, golds: list) -> None:
    a, b = extract_v1(text), extract_v2(text)
    _assert_well_formed(text, a)
    _assert_well_formed(text, b)
    for ex in (a, b):  # marker and bold results come only from qualifying content
        if ex.method == "hash":
            assert is_hash_answer(ex.content) and _spec_is_hash_answer(ex.content), (text, ex)
        if ex.method == "bold":
            assert BOLD_NUMERIC_RE.match(ex.content) and SPEC_BOLD_NUMERIC_RE.match(ex.content), (text, ex)
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
    _assert_monotone(text, [gold])  # also checks the result contract (incl. .extracted) for both


_DIGIT_FREE_HEADINGS = (
    "\n#### Final Answer",
    "\n#### Verification\n",
    "\n####",
    "\n#### \n",
    "\n### Final Answer",
    "\n#### ...",
    "\n#### .\n",
    "\n#### -.",
    "\n#### $.",
)


@given(RESPONSES, st.integers(0, 12))
@settings(derandomize=True, database=None, max_examples=500, suppress_health_check=[HealthCheck.too_slow])
def test_trailing_headings_never_change_the_answer(text: str, k: int) -> None:
    """A heading (incl. numbered) or numberless "####" line appended to a response is never a marker: v1 is
    unchanged by any of them, v2 by any digit-free one (digits may only feed v2's last-number stage)."""
    a, b = extract_v1(text), extract_v2(text)
    with_digits = (
        f"\n#### Step {k}: Verify",
        f"\n#### Step {k}\n",
        f"\n### Step {k}",
        f"\n**Step {k}: Add**",
        f"\n#### {k}. Verification",
        f"\n#### {k}) Check the answer\n",
    )
    for suffix in (*_DIGIT_FREE_HEADINGS, *with_digits):
        assert extract_v1(text + suffix) == a, suffix
    for suffix in _DIGIT_FREE_HEADINGS:
        assert extract_v2(text + suffix) == b, suffix


_TITLE = st.from_regex(r"[A-Za-z][A-Za-z0-9 :,()-]{0,20}", fullmatch=True)
_BLANKS = st.text(alphabet=" \t\u00a0\u2003", min_size=1, max_size=4)


@given(st.integers(0, 10**6), st.sampled_from([".", ")"]), _BLANKS, _TITLE, RESPONSES)
@settings(derandomize=True, database=None, max_examples=500, suppress_health_check=[HealthCheck.too_slow])
def test_numbered_headings_never_qualify(n: int, punct: str, blanks: str, title: str, text: str) -> None:
    """Any "<n>. <Title>" / "<n>) <Title>" line is not a marker for either extractor; numbers still are."""
    heading = f"{n}{punct}{blanks}{title}"
    assert not is_hash_answer(heading) and not _spec_is_hash_answer(heading)
    for number in (f"{n}", f"{n}.", f"{n}.5", f"{n}{punct}", f".{n}"):
        assert is_hash_answer(number) and _spec_is_hash_answer(number), number
    assert extract_v1(f"#### {n}.").value == n  # "#### 4." reads 4
    a, b = extract_v1(text), extract_v2(text)
    assert extract_v1(f"{text}\n#### {heading}") == a
    if b.method in ("boxed", "hash"):  # v2's marker stage is unaffected (later stages may read its digits)
        assert extract_v2(f"{text}\n#### {heading}") == b


def test_parse_lenient_agrees_with_strict_numbers_for_every_unicode_digit() -> None:
    """v1's ``\\d`` is Unicode, so the monotonicity lemma must hold for every Nd code point, not only ASCII
    (NFKC rewrites some digits, e.g. full-width / mathematical ones, and leaves others alone)."""
    digit = re.compile(r"\d")
    seen = 0
    for cp in range(sys.maxunicode + 1):
        c = chr(cp)
        if not digit.fullmatch(c):
            continue
        seen += 1
        s = f"-{c}7.{c}"
        assert parse_lenient(s) == Fraction(Decimal(s)), hex(cp)
        text = f"\\boxed{{{s}}}"
        a, b = extract_v1(text), extract_v2(text)
        assert a.value is not None and (b.value, b.method, b.span) == (a.value, a.method, a.span), hex(cp)
    assert seen >= 600  # ~660 decimal digits in Unicode 14/15


@given(st.from_regex(r"-?\d+(?:\.\d+)?", fullmatch=True))
@settings(derandomize=True, database=None, max_examples=500)
def test_parse_lenient_agrees_with_strict_numbers(s: str) -> None:
    """The lemma behind monotonicity: parse_lenient(s) == Fraction(Decimal(s)) for every v1-valid s."""
    assert parse_lenient(s) == Fraction(Decimal(s))
    assert parse_lenient(s + ".") == Fraction(Decimal(s))
