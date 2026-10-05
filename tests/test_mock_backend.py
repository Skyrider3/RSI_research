"""Tests for the deterministic synthetic backend (driftlab.backends.mock).

Answer formats are checked with simple local regexes on purpose: the extraction package is developed
independently and must not be a dependency of these tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from driftlab.backends.base import Backend, GenRequest, GenResult
from driftlab.backends.mock import (
    FORMATS,
    MOCK_VERSION,
    SNIPPETS,
    MockBackend,
    answer_key_from_jsonl,
    prompt_features,
    prompt_quality,
)
from driftlab.config import MockSection
from driftlab.environments import Decoding
from driftlab.keys import engine_fingerprint, proposer_seed, sample_seed
from driftlab.prompting import PROPOSER_SYSTEM, fill, load_prompt_file

DATA = Path(__file__).resolve().parents[1] / "data" / "gsm8k" / "test_first200.jsonl"
MODEL, REV = "Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

GREEDY = Decoding(id="greedy", temperature=0.0)
T02 = Decoding(id="t02", temperature=0.2)
PROPOSER = Decoding(id="proposer", temperature=0.7)

INITIAL = load_prompt_file("initial_v1.txt")
GOOD = (
    "You are a helpful assistant that solves grade-school math word problems. Think step by step and show "
    "your work. Double-check each arithmetic step before answering. Read the question carefully and identify "
    "what is asked. Put your final answer within \\boxed{}."
)
BRIEF = (
    "You are a helpful assistant that solves grade-school math word problems. Answer as briefly as possible. "
    "Put your final answer within \\boxed{}."
)
PLAIN = (
    INITIAL + " Give the final answer as a plain number with no commas, units, or currency symbols inside "
    "\\boxed{}."
)
UNITS = INITIAL + " State units in the final answer."
NO_BOX = (
    "You are a helpful assistant that solves grade-school math word problems. Solve the problem step by step."
)

# --------------------------------------------------------------------------- local format checks

BOX_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
HASH_RE = re.compile(r"^####\s*(.*?)\s*$", re.M)
STRICT_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
TEXT_RE = re.compile(r"\\text\{[^{}]*\}")


def _last_marker(text: str) -> tuple[str, str] | None:
    """(kind, content) of whichever of the last box / last #### line starts later."""
    boxes, hashes = list(BOX_RE.finditer(text)), list(HASH_RE.finditer(text))
    box, hsh = (boxes[-1] if boxes else None), (hashes[-1] if hashes else None)
    if box is not None and (hsh is None or box.start() > hsh.start()):
        return "box", box.group(1).strip()
    if hsh is not None:
        return "hash", hsh.group(1).rstrip(".").strip()
    return None


def strict_value(text: str) -> str | None:
    marker = _last_marker(text)
    if marker is None or not STRICT_NUM_RE.fullmatch(marker[1]):
        return None
    return marker[1]


def lenient_value(text: str) -> str | None:
    marker = _last_marker(text)
    if marker is not None:
        content = TEXT_RE.sub("", marker[1]).replace("\\$", "").replace("$", "")
        m = NUM_RE.search(content)
        return m.group().replace(",", "") if m else None
    for rx in (r"answer is\s*\$?(-?[\d,.]*\d)", r"\*\*\s*(-?[\d,.]*\d)\s*\*\*"):
        found = re.findall(rx, text)
        if found:
            return found[-1].replace(",", "")
    return None


def same_number(a: str | None, b: str) -> bool:
    return a is not None and float(a) == float(b)


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def answer_key() -> dict[str, str]:
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["question"]: r["gold"] for r in rows}


@pytest.fixture(scope="module")
def questions(answer_key: dict[str, str]) -> list[str]:
    return list(answer_key)


@pytest.fixture(scope="module")
def backend(answer_key: dict[str, str]) -> MockBackend:
    return MockBackend(MODEL, REV, answer_key=answer_key)


def _run(b: MockBackend, system: str, qs: list[str], dec: Decoding = GREEDY, **kw) -> list[GenResult]:
    return b.generate([GenRequest(system, q, dec, **kw) for q in qs])


def _sampled(b: MockBackend, system: str, qs: list[str], slot: int = 0, draw: int = 0) -> list[GenResult]:
    reqs = [
        GenRequest(system, q, T02, seed=sample_seed(0, "eval", "t02", slot, "round", draw, "test", n))
        for n, q in enumerate(qs)
    ]
    return b.generate(reqs)


def _accuracy(results: list[GenResult], qs: list[str], key: dict[str, str], parser=lenient_value) -> float:
    return sum(same_number(parser(r.text), key[q]) for r, q in zip(results, qs, strict=True)) / len(qs)


def _meta_prompt(current: str = INITIAL, golds: tuple[str, ...] = ("18", "1250", "7", "42", "360")) -> str:
    block = load_prompt_file("error_block_v1.txt")
    blocks = "\n\n".join(
        fill(block, i=i + 1, question=f"Problem {i} with $4 and 3 apples.", response="...", gold=g)
        for i, g in enumerate(golds)
    )
    return fill(
        load_prompt_file("meta_v1.txt"), current_prompt=current, n_errors=len(golds), error_blocks=blocks
    )


PROMPT_TAG_RE = re.compile(r"<prompt>(.*?)</prompt>", re.S)


def _classify_proposal(text: str, shown_golds: tuple[str, ...] = ("1250", "360")) -> str:
    tags = PROMPT_TAG_RE.findall(text)
    if not tags:
        return "missing_tags"
    body = tags[-1].strip()
    if not body:
        return "empty_tags"
    if "<prompt>" in body:
        return "nested"
    if "\\boxed" not in body:
        return "no_boxed"
    if any(re.search(rf"(?<!\d){g}(?!\d)", body) for g in shown_golds):
        return "copies_gold"
    return "ok"


# --------------------------------------------------------------------------- protocol


def test_protocol_attributes_and_render() -> None:
    b = MockBackend(MODEL, REV)
    assert isinstance(b, Backend)
    assert b.kind == "mock" and b.synthetic is True
    assert b.model_id == MODEL and b.model_revision == REV
    assert b.render("SYS", "USER") == (
        "<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nUSER<|im_end|>\n<|im_start|>assistant\n"
    )


def test_engine_info_tracks_config() -> None:
    info = MockBackend(MODEL, REV).engine_info()
    assert info["kind"] == "mock" and info["version"] == MOCK_VERSION == "1"
    defaults = MockSection()
    for name in ("base_quality", "sample_sd", "greedy_flip_rate", "malformed_proposal_rate"):
        assert info[name] == getattr(defaults, name)
    json.dumps(info)  # serialisable (stored in provenance / fingerprints)
    fp = engine_fingerprint(info)
    assert engine_fingerprint(MockBackend(MODEL, REV).engine_info()) == fp
    assert engine_fingerprint(MockBackend(MODEL, REV, MockSection(greedy_flip_rate=0.01)).engine_info()) != fp
    assert engine_fingerprint(MockBackend(MODEL, REV, MockSection(base_quality=0.3)).engine_info()) != fp
    assert engine_fingerprint(MockBackend(MODEL, REV, answer_key={"q": "1"}).engine_info()) != fp


def test_result_fields(backend: MockBackend, questions: list[str]) -> None:
    res = _run(backend, INITIAL, questions[:20])
    for r, q in zip(res, questions[:20], strict=True):
        assert r.finish_reason in {"stop", "length"}
        assert r.text
        words = len(INITIAL.split()) + len(q.split())
        assert abs(r.n_prompt_tokens - 1.3 * words) <= 1
        if r.finish_reason == "stop":
            assert abs(r.n_completion_tokens - 1.3 * len(r.text.split())) <= 1
        assert 0 < r.n_completion_tokens <= 640
        assert 0 < r.latency_ms < 100
    assert [r.latency_ms for r in res] == [r.latency_ms for r in _run(backend, INITIAL, questions[:20])]


# --------------------------------------------------------------------------- determinism


def test_determinism_and_order_independence(answer_key: dict[str, str], questions: list[str]) -> None:
    qs = questions[:30]
    reqs = (
        [GenRequest(INITIAL, q, GREEDY) for q in qs]
        + [GenRequest(GOOD, q, T02, seed=n) for n, q in enumerate(qs)]
        + [GenRequest(INITIAL, q, GREEDY, nonce="rerun:3") for q in qs]
        + [
            GenRequest(PROPOSER_SYSTEM, _meta_prompt(), PROPOSER, seed=proposer_seed(0, t, 0))
            for t in range(5)
        ]
    )
    cfg = MockSection(greedy_flip_rate=0.2)
    b1, b2 = MockBackend(MODEL, REV, cfg, answer_key), MockBackend(MODEL, REV, cfg, answer_key)
    ref = b1.generate(reqs)
    assert ref == b1.generate(reqs)  # repeatable
    assert ref == b2.generate(reqs)  # no hidden instance state
    order = list(reversed(range(len(reqs))))
    order = order[::2] + order[1::2]
    shuffled = b2.generate([reqs[i] for i in order])
    assert {i: r for i, r in zip(order, shuffled, strict=True)} == dict(enumerate(ref))
    singles = [b2.generate([r])[0] for r in reqs]  # batch composition does not matter
    assert singles == ref


def test_decoding_id_is_only_a_label(backend: MockBackend, questions: list[str]) -> None:
    other = Decoding(id="something_else", temperature=0.0)
    assert _run(backend, INITIAL, questions[:20]) == _run(backend, INITIAL, questions[:20], other)


def test_greedy_ignores_seed(backend: MockBackend, questions: list[str]) -> None:
    qs = questions[:40]
    base = _run(backend, INITIAL, qs)
    assert base == _run(backend, INITIAL, qs, seed=123) == _run(backend, INITIAL, qs, seed=99999)


def test_sampling_varies_with_seed(
    backend: MockBackend, answer_key: dict[str, str], questions: list[str]
) -> None:
    q = questions[0]
    texts = {r.text for r in backend.generate([GenRequest(INITIAL, q, T02, seed=s) for s in range(20)])}
    assert len(texts) >= 10
    same = backend.generate([GenRequest(INITIAL, q, T02, seed=7)] * 2)
    assert same[0] == same[1]
    # correctness itself varies across seeds on some items, and sampling differs from greedy on a minority
    a, b = _sampled(backend, INITIAL, questions, draw=0), _sampled(backend, INITIAL, questions, draw=1)
    ca = [same_number(lenient_value(r.text), answer_key[q]) for r, q in zip(a, questions, strict=True)]
    cb = [same_number(lenient_value(r.text), answer_key[q]) for r, q in zip(b, questions, strict=True)]
    assert 0.02 <= sum(x != y for x, y in zip(ca, cb, strict=True)) / len(questions) <= 0.35
    g = _run(backend, INITIAL, questions)
    cg = [same_number(lenient_value(r.text), answer_key[q]) for r, q in zip(g, questions, strict=True)]
    assert 0.02 <= sum(x != y for x, y in zip(ca, cg, strict=True)) / len(questions) <= 0.35


def test_sampling_noise_scales_with_temperature(answer_key: dict[str, str], questions: list[str]) -> None:
    b = MockBackend(MODEL, REV, answer_key=answer_key)
    greedy = [b.explain(GenRequest(INITIAL, q, GREEDY))["solved"] for q in questions]

    def flips(temp: float) -> int:
        dec = Decoding(id=f"t{temp}", temperature=temp)
        return sum(
            b.explain(GenRequest(INITIAL, q, dec, seed=n))["solved"] != g
            for n, (q, g) in enumerate(zip(questions, greedy, strict=True))
        )

    assert flips(0.05) < flips(0.2) < flips(1.0)


# --------------------------------------------------------------------------- physical greedy reruns


def test_greedy_rerun_identical_without_flip_rate(backend: MockBackend, questions: list[str]) -> None:
    base = _run(backend, INITIAL, questions)
    for nonce in ("rerun:1", "rerun:5", "audit:0"):
        assert _run(backend, INITIAL, questions, nonce=nonce) == base


def test_greedy_rerun_flips_with_positive_rate(answer_key: dict[str, str], questions: list[str]) -> None:
    b = MockBackend(MODEL, REV, MockSection(greedy_flip_rate=0.05), answer_key)
    base = _run(b, INITIAL, questions)
    assert base == _run(
        MockBackend(MODEL, REV, answer_key=answer_key), INITIAL, questions
    )  # creation unaffected
    n_diff = n_flip = n = 0
    for r in range(1, 6):
        rerun = _run(b, INITIAL, questions, nonce=f"rerun:{r}")
        assert rerun == _run(b, INITIAL, questions, nonce=f"rerun:{r}")  # the same nonce is reproducible
        n_diff += sum(x.text != y.text for x, y in zip(base, rerun, strict=True))
        n_flip += sum(
            b.explain(GenRequest(INITIAL, q, GREEDY, nonce=f"rerun:{r}"))["flipped"] for q in questions
        )
        n += len(questions)
    assert 0.02 <= n_flip / n <= 0.09  # ~ greedy_flip_rate
    assert n_flip < n_diff <= 4 * n_flip  # some reruns change only the wording
    # a flipped item really changes its *extracted* correctness; a wording-only divergence never changes the
    # extracted answer; every other rerun is byte-identical to the creation generation
    rerun = _run(b, INITIAL, questions, nonce="rerun:1")
    seen: Counter[str] = Counter()
    for q, x, y in zip(questions, base, rerun, strict=True):
        info = b.explain(GenRequest(INITIAL, q, GREEDY, nonce="rerun:1"))
        if not (info["flipped"] or info["diverged"]):
            assert x == y
            continue
        assert x.text != y.text
        if "length" in (x.finish_reason, y.finish_reason):
            continue
        before = same_number(lenient_value(x.text), answer_key[q])
        after = same_number(lenient_value(y.text), answer_key[q])
        if info["flipped"]:
            seen["flipped"] += 1
            assert info["solved"] != b.explain(GenRequest(INITIAL, q, GREEDY))["solved"]
            assert before != after
        else:
            seen["diverged"] += 1
            assert lenient_value(x.text) == lenient_value(y.text) and before == after
    assert seen["flipped"] > 0 and seen["diverged"] > 0


def test_flip_rate_does_not_touch_sampling_or_non_nonce(
    answer_key: dict[str, str], questions: list[str]
) -> None:
    qs = questions[:50]
    b0 = MockBackend(MODEL, REV, answer_key=answer_key)
    b1 = MockBackend(MODEL, REV, MockSection(greedy_flip_rate=0.5), answer_key)
    assert _sampled(b0, INITIAL, qs) == _sampled(b1, INITIAL, qs)
    assert _run(b0, INITIAL, qs) == _run(b1, INITIAL, qs)


# --------------------------------------------------------------------------- prompt quality


def test_prompt_features() -> None:
    f = prompt_features(INITIAL)
    assert f.step and f.boxed and not (f.brief or f.plain or f.units_format or f.verify)
    assert prompt_features(PLAIN).plain and prompt_features(UNITS).units_format
    assert not prompt_features(UNITS).units_track
    assert prompt_features(BRIEF).brief and not prompt_features(BRIEF).step
    assert prompt_features(GOOD).verify and prompt_features(GOOD).read
    assert prompt_features("Keep track of units and convert them when needed.").units_track
    assert not prompt_features("Do not answer briefly; think step by step.").brief
    assert prompt_features("Do not answer briefly; think step by step.").step
    assert prompt_features(" ".join(["word"] * 181)).long
    assert not prompt_features(NO_BOX).boxed


def test_prompt_quality_jitter_bounded() -> None:
    for p in (INITIAL, GOOD, BRIEF, PLAIN, UNITS, NO_BOX):
        f = prompt_features(p)
        q0 = 0.55 + 0.25 * f.step + 0.15 * f.verify + 0.10 * f.read + 0.05 * f.units_track
        q0 += -0.3 * f.brief - 0.3 * f.intuition - 0.1 * f.long
        assert abs(prompt_quality(p) - q0) <= 0.15 + 1e-12
    b = MockBackend(MODEL, REV, MockSection(base_quality=0.3))
    assert b.quality(INITIAL) == pytest.approx(prompt_quality(INITIAL) - 0.25)


def test_good_prompt_beats_brief_prompt(
    backend: MockBackend, answer_key: dict[str, str], questions: list[str]
) -> None:
    good = _accuracy(_run(backend, GOOD, questions), questions, answer_key)
    brief = _accuracy(_run(backend, BRIEF, questions), questions, answer_key)
    base = _accuracy(_run(backend, INITIAL, questions), questions, answer_key)
    assert good > brief + 0.10
    assert 0.55 <= base <= 0.80  # initial prompt, lenient parsing
    assert brief < base < good + 0.02
    # same ordering under sampling
    good_s = _accuracy(_sampled(backend, GOOD, questions), questions, answer_key)
    brief_s = _accuracy(_sampled(backend, BRIEF, questions), questions, answer_key)
    assert good_s > brief_s + 0.10


def test_gold_used_for_correct_answers(
    backend: MockBackend, answer_key: dict[str, str], questions: list[str]
) -> None:
    for q in questions[:60]:
        req = GenRequest(INITIAL, q, GREEDY)
        info = backend.explain(req)
        assert info["gold"] == answer_key[q] == backend.gold_for(q)
        if info["solved"]:
            assert info["value"] == answer_key[q]
        else:
            assert float(info["value"]) != float(answer_key[q])
        text = backend.generate([req])[0].text
        if info["format"] != "truncated":
            assert same_number(lenient_value(text), info["value"])


def test_unknown_question_gets_pseudo_gold() -> None:
    b = MockBackend(MODEL, REV)
    q = "Tom has 3 boxes with 12 pencils each. How many pencils does he have?"
    gold = b.gold_for(q)
    assert re.fullmatch(r"\d+", gold) and int(gold) >= 2
    assert MockBackend(MODEL, REV).gold_for(q) == gold
    res = b.generate([GenRequest(INITIAL, q, GREEDY)])
    assert res == MockBackend(MODEL, REV).generate([GenRequest(INITIAL, q, GREEDY)])
    assert MockBackend(MODEL, REV, answer_key={q: "36"}).gold_for(q) == "36"


def test_answer_key_accepts_numbers_and_fractions() -> None:
    q1, q2, q3 = "How many apples are left?", "How much does it cost?", "What is the total?"
    b = MockBackend(MODEL, REV, answer_key={q1: Fraction(18), q2: Fraction(3, 2), q3: "1,234"})  # type: ignore[dict-item]
    assert (b.gold_for(q1), b.gold_for(q2), b.gold_for(q3)) == ("18", "1.5", "1234")


def test_answer_key_from_jsonl(answer_key: dict[str, str]) -> None:
    assert answer_key_from_jsonl(DATA) == answer_key
    assert len(answer_key) == 200


def test_answer_key_from_jsonl_falls_back_to_answer(tmp_path: Path) -> None:
    rows = [
        {"question": "Q1?", "answer": "3 + 4 = 7\n#### 7"},
        {"question": "Q2?", "answer": "so 1,234 in total\n#### 1,234"},
        {"question": "Q3?", "answer": "x\n#### 5", "gold": "5"},
        {"question": "Q4?", "answer": "x\n#### 9", "gold": None},
    ]
    path = tmp_path / "split.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    assert answer_key_from_jsonl(path) == {"Q1?": "7", "Q2?": "1234", "Q3?": "5", "Q4?": "9"}


def test_non_numeric_answer_key_value_raises() -> None:
    """Regression: a non-numeric gold silently became a pseudo-gold (the item could never be answered right)."""
    for bad in ("N/A", "", "nan", None):
        with pytest.raises(ValueError, match="non-numeric"):
            MockBackend(MODEL, REV, answer_key={"How many apples?": bad})  # type: ignore[dict-item]


def test_solved_answers_equal_decimal_golds() -> None:
    """Regression: terminating decimal golds were printed rounded to 2 decimals (2.125 -> 2.12), so a solved
    item was scored wrong by every extractor."""
    golds = ("2.125", "-1.5", "0.0009765625", "12345.678")
    key = {f"Item {i}: a shop sells {i} pens. How much is it?": golds[i % 4] for i in range(80)}
    b = MockBackend(MODEL, REV, answer_key=key)
    n_solved = 0
    for q, g in key.items():
        assert b.gold_for(q) == g
        req = GenRequest(INITIAL, q, GREEDY)
        info, text = b.explain(req), b.generate([req])[0].text
        if info["solved"] and info["format"] != "truncated":
            n_solved += 1
            assert info["value"] == g and same_number(lenient_value(text), g)
            if info["format"] == "box":
                assert same_number(strict_value(text), g)
    assert n_solved >= 20


def test_config_is_copied(answer_key: dict[str, str], questions: list[str]) -> None:
    """Regression: the backend aliased the caller's MockSection, so mutating it after the engine had
    fingerprinted engine_info() changed outputs, and only for prompts not yet in the quality cache."""
    cfg = MockSection(greedy_flip_rate=0.1)
    b = MockBackend(MODEL, REV, cfg, answer_key)
    info = b.engine_info()
    reqs = [GenRequest(p, q, GREEDY, nonce="rerun:1") for p in (INITIAL, GOOD) for q in questions[:30]]
    before = b.generate(reqs[:30])  # caches the quality of INITIAL only
    cfg.base_quality, cfg.greedy_flip_rate, cfg.malformed_proposal_rate = -5.0, 0.0, 1.0
    fresh = MockBackend(MODEL, REV, MockSection(greedy_flip_rate=0.1), answer_key)
    assert b.engine_info() == info == fresh.engine_info()
    assert b.generate(reqs) == fresh.generate(reqs) and before == fresh.generate(reqs[:30])
    assert b.quality(GOOD) == fresh.quality(GOOD)


def test_empty_batch() -> None:
    assert MockBackend(MODEL, REV).generate([]) == []


# --------------------------------------------------------------------------- answer formats


def _strict_compliance(results: list[GenResult]) -> float:
    done = [r for r in results if r.finish_reason == "stop"]
    return sum(strict_value(r.text) is not None for r in done) / len(done)


def test_format_instructions_drive_strict_compliance(backend: MockBackend, questions: list[str]) -> None:
    base = _strict_compliance(_run(backend, INITIAL, questions))
    plain = _strict_compliance(_run(backend, PLAIN, questions))
    units = _strict_compliance(_run(backend, UNITS, questions))
    no_box = _strict_compliance(_run(backend, NO_BOX, questions))
    assert 0.75 <= base <= 0.93
    assert plain >= 0.92
    assert plain > base > units
    assert units <= 0.75
    assert no_box < 0.5  # without a \boxed instruction, unboxed formats dominate
    # and the strict parser therefore under-counts accuracy relative to lenient parsing
    sampled_plain = _strict_compliance(_sampled(backend, PLAIN, questions))
    sampled_units = _strict_compliance(_sampled(backend, UNITS, questions))
    assert sampled_plain > sampled_units + 0.2


def test_strict_accuracy_below_lenient(
    backend: MockBackend, answer_key: dict[str, str], questions: list[str]
) -> None:
    res = _run(backend, INITIAL, questions)
    strict = _accuracy(res, questions, answer_key, parser=strict_value)
    lenient = _accuracy(res, questions, answer_key)
    assert lenient - 0.25 <= strict < lenient - 0.03
    for r, q in zip(res, questions, strict=True):  # monotone: strict-correct implies lenient-correct
        if same_number(strict_value(r.text), answer_key[q]):
            assert same_number(lenient_value(r.text), answer_key[q])


def test_all_formats_occur(backend: MockBackend, questions: list[str]) -> None:
    seen: Counter[str] = Counter()
    for p in (INITIAL, UNITS, NO_BOX, GOOD):
        for dec in (GREEDY, T02):
            for n, q in enumerate(questions):
                seen[backend.explain(GenRequest(p, q, dec, seed=n))["format"]] += 1
    assert set(seen) == set(FORMATS)


def test_format_shapes(backend: MockBackend, questions: list[str]) -> None:
    pattern = {
        "box": re.compile(r"\\boxed\{-?\d+(?:\.\d+)?\}"),
        "comma": re.compile(r"\\boxed\{-?\d{1,3}(?:,\d{3})+(?:\.\d+)?\}"),
        "currency": re.compile(r"\\boxed\{\\\$-?\d+(?:\.\d+)?\}"),
        "units": re.compile(r"\\boxed\{-?\d+(?:\.\d+)? \\text\{ [a-z]+\}\}"),
        "answer_is": re.compile(r"The answer is -?\d+(?:\.\d+)?\.$"),
        "bold": re.compile(r"\*\*-?\d+(?:\.\d+)?\*\*$"),
        "hash": re.compile(r"^#### -?\d+(?:\.\d+)?$", re.M),
    }
    for p in (INITIAL, UNITS, NO_BOX):
        for n, q in enumerate(questions):
            req = GenRequest(p, q, T02, seed=n)
            fmt = backend.explain(req)["format"]
            text = backend.generate([req])[0].text
            lines = text.split("\n")
            if fmt == "truncated":
                continue
            assert 4 <= len(lines) <= 9  # 3-8 reasoning lines + the final-answer line
            assert pattern[fmt].search(lines[-1]), (fmt, lines[-1])
            if fmt == "comma":
                assert float(backend.explain(req)["value"]) >= 1000
            assert "\\boxed" not in "\n".join(lines[:-1]) and "####" not in "\n".join(lines[:-1])


def test_commas_only_for_large_values(backend: MockBackend, questions: list[str]) -> None:
    n_comma = 0
    for r in _run(backend, INITIAL, questions) + _sampled(backend, INITIAL, questions):
        for content in BOX_RE.findall(r.text):
            if "," in content:
                n_comma += 1
                assert float(content.replace(",", "")) >= 1000
    assert n_comma > 0


def test_currency_more_likely_for_dollar_questions(backend: MockBackend, questions: list[str]) -> None:
    formats = {
        q: backend.explain(GenRequest(INITIAL, q, T02, seed=n))["format"] for n, q in enumerate(questions)
    }
    dollar = [q for q in questions if "$" in q]
    assert sum(formats[q] == "currency" for q in dollar) > 0
    assert all(formats[q] != "currency" for q in questions if "$" not in q)


def test_truncation_present_but_rare(backend: MockBackend, questions: list[str]) -> None:
    results = []
    for p in (INITIAL, GOOD, PLAIN, UNITS):
        results += _run(backend, p, questions) + _sampled(backend, p, questions)
    trunc = [r for r in results if r.finish_reason == "length"]
    assert 0 < len(trunc) <= 0.04 * len(results)
    for r in trunc:
        assert "\\boxed" not in r.text and "####" not in r.text and "answer is" not in r.text
        assert r.n_completion_tokens == 640
        assert len(r.text.split()) > 300  # cut at the token cap, mid-reasoning


def test_max_new_tokens_is_honoured(backend: MockBackend, questions: list[str]) -> None:
    short = Decoding(id="short", temperature=0.0, max_new_tokens=12)
    for r in _run(backend, INITIAL, questions[:20], short):
        assert r.finish_reason == "length"
        assert r.n_completion_tokens <= 12
        assert len(r.text.split()) <= 12 / 1.3


# --------------------------------------------------------------------------- proposer mode


def test_proposer_returns_tagged_candidates() -> None:
    b = MockBackend(MODEL, REV)
    meta = _meta_prompt()
    res = b.generate(
        [GenRequest(PROPOSER_SYSTEM, meta, PROPOSER, seed=proposer_seed(0, t, 0)) for t in range(200)]
    )
    kinds = Counter(_classify_proposal(r.text) for r in res)
    assert kinds["ok"] >= 0.8 * len(res)
    ok = [PROMPT_TAG_RE.findall(r.text)[-1].strip() for r in res if _classify_proposal(r.text) == "ok"]
    assert all(c != INITIAL for c in ok)
    assert all(40 <= len(c) <= 1500 for c in ok)
    assert len(set(ok)) >= 30  # different seeds -> different candidates
    assert all(r.finish_reason == "stop" and r.n_completion_tokens > 0 for r in res)
    # candidates are built from the snippet pool: some help, some hurt
    qualities = [b.quality(c) for c in ok]
    assert min(qualities) < b.quality(INITIAL) < max(qualities)
    snippet_texts = [s.text for s in SNIPPETS]
    assert sum(any(s in c for s in snippet_texts) for c in ok) >= 0.8 * len(ok)


def test_proposer_edits_the_incumbent() -> None:
    b = MockBackend(MODEL, REV, MockSection(malformed_proposal_rate=0.0))
    incumbent = INITIAL + " Double-check each arithmetic step before answering."
    res = b.generate(
        [GenRequest(PROPOSER_SYSTEM, _meta_prompt(incumbent), PROPOSER, seed=s) for s in range(50)]
    )
    for r in res:
        cand = PROMPT_TAG_RE.findall(r.text)[-1].strip()
        assert cand != incumbent and "\\boxed" in cand
        overlap = set(re.split(r"(?<=\.)\s+", incumbent)) & set(re.split(r"(?<=\.)\s+", cand))
        assert overlap  # an edit of the incumbent, not a fresh prompt


def test_proposer_determinism() -> None:
    b = MockBackend(MODEL, REV)
    meta = _meta_prompt()
    greedy = Decoding(id="proposer0", temperature=0.0)
    g = b.generate([GenRequest(PROPOSER_SYSTEM, meta, greedy, seed=s) for s in range(5)])
    assert len({r.text for r in g}) == 1  # temperature 0 ignores the seed
    s1 = b.generate([GenRequest(PROPOSER_SYSTEM, meta, PROPOSER, seed=11)])
    assert s1 == MockBackend(MODEL, REV).generate([GenRequest(PROPOSER_SYSTEM, meta, PROPOSER, seed=11)])


@pytest.mark.parametrize("rate", [0.0, 0.1, 0.3, 1.0])
def test_malformed_rate(rate: float) -> None:
    b = MockBackend(MODEL, REV, MockSection(malformed_proposal_rate=rate))
    meta = _meta_prompt()
    n = 600
    res = b.generate([GenRequest(PROPOSER_SYSTEM, meta, PROPOSER, seed=s) for s in range(n)])
    kinds = Counter(_classify_proposal(r.text) for r in res)
    bad = (n - kinds["ok"]) / n
    if rate in (0.0, 1.0):
        assert bad == rate  # the extremes are exact: never / always malformed
    else:
        assert abs(bad - rate) <= 0.05
    for s, r in enumerate(res):  # explain() reports exactly the malformation visible in the text
        info = b.explain(GenRequest(PROPOSER_SYSTEM, meta, PROPOSER, seed=s))
        assert info["mode"] == "proposer" and info["incumbent_found"] is True
        assert (info["malformed"] or "ok") == _classify_proposal(r.text)
    if rate >= 0.3:  # every malformation type is exercised
        assert {"missing_tags", "empty_tags", "nested", "no_boxed", "copies_gold"} <= set(kinds)


def test_proposer_without_large_golds_never_copies() -> None:
    b = MockBackend(MODEL, REV, MockSection(malformed_proposal_rate=1.0))
    meta = _meta_prompt(golds=("18", "7", "42", "5", "12"))
    res = b.generate([GenRequest(PROPOSER_SYSTEM, meta, PROPOSER, seed=s) for s in range(100)])
    kinds = Counter(_classify_proposal(r.text, shown_golds=()) for r in res)
    assert kinds["ok"] == 0 and "copies_gold" not in kinds


def test_proposer_handles_missing_incumbent() -> None:
    b = MockBackend(MODEL, REV, MockSection(malformed_proposal_rate=0.0))
    r = b.generate([GenRequest(PROPOSER_SYSTEM, "Improve the prompt.", PROPOSER, seed=1)])[0]
    assert "\\boxed" in PROMPT_TAG_RE.findall(r.text)[-1]


# --------------------------------------------------------------------------- frozen behaviour

# Inline on purpose (not the packaged templates): the pin must move only when the mock itself changes.
_PIN_PROMPTS = (
    "You are a helpful assistant that solves grade-school math word problems. Solve the problem step by step, "
    "and put your final answer within \\boxed{}.",
    "Answer as briefly as possible. State units in the final answer. Put the answer in \\boxed{}.",
    "You solve math word problems.",
)
_PIN_META = (
    f"<current_prompt>\n{_PIN_PROMPTS[0]}\n</current_prompt>\n\n### Problem 1\nQ\n\n"
    "### Correct final answer\n1250\n"
)
# MOCK_VERSION -> sha256 of _pinned_digest(). Add a new entry (never edit an old one) when the mock changes.
PINNED_OUTPUT_DIGESTS = {"1": "36299004c086c52a7fbfd96150f91d70e5feac5b16003dde51bf008c71f804b9"}


def _pinned_digest() -> str:
    """sha256 of the mock's outputs on a fixed request set covering every answer-mode and proposer path."""
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines()[:40]]
    key = {r["question"]: r["gold"] for r in rows}
    cfg = MockSection(base_quality=0.55, sample_sd=0.6, greedy_flip_rate=0.2, malformed_proposal_rate=0.3)
    b = MockBackend(MODEL, REV, cfg, key)
    reqs = [
        req
        for p in _PIN_PROMPTS
        for n, q in enumerate(key)
        for req in (
            GenRequest(p, q, GREEDY),
            GenRequest(p, q, T02, seed=n),
            GenRequest(p, q, GREEDY, nonce="r:1"),
        )
    ]
    reqs += [GenRequest(PROPOSER_SYSTEM, _PIN_META, PROPOSER, seed=s) for s in range(40)]
    h = hashlib.sha256()
    for r in b.generate(reqs):
        h.update(
            json.dumps(
                [r.text, r.finish_reason, r.n_prompt_tokens, r.n_completion_tokens, r.latency_ms]
            ).encode()
        )
    return h.hexdigest()


def test_outputs_pinned_to_mock_version() -> None:
    """Stored generations are found by gen_key, whose engine fingerprint covers engine_info() (and so
    MOCK_VERSION) but not the mock's code: any change to generated outputs must bump MOCK_VERSION, otherwise
    a resumed run silently mixes generations of two different synthetic models."""
    assert _pinned_digest() == PINNED_OUTPUT_DIGESTS.get(MOCK_VERSION), (
        "mock outputs changed: bump MOCK_VERSION and pin the new digest"
    )


def test_outputs_independent_of_hash_seed() -> None:
    """Resume across processes: outputs must not depend on PYTHONHASHSEED (set/dict-order randomness)."""
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('tmb', {str(Path(__file__))!r})\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['tmb'] = m; spec.loader.exec_module(m)\n"
        "print(m._pinned_digest())\n"
    )
    want = _pinned_digest()
    for hash_seed in ("0", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        out = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == want


# --------------------------------------------------------------------------- property test


@settings(max_examples=60, deadline=None)
@given(
    system=st.text(min_size=0, max_size=200),
    user=st.text(min_size=0, max_size=200),
    temp=st.sampled_from([0.0, 0.2, 0.7]),
    seed=st.one_of(st.none(), st.integers(min_value=0, max_value=2**31 - 1)),
    nonce=st.one_of(st.none(), st.just("rerun:2")),
)
def test_any_request_is_handled(
    system: str, user: str, temp: float, seed: int | None, nonce: str | None
) -> None:
    b = MockBackend(MODEL, REV, MockSection(greedy_flip_rate=0.3))
    req = GenRequest(system, user, Decoding(id="d", temperature=temp), seed=seed, nonce=nonce)
    (res,) = b.generate([req])
    assert res.finish_reason in {"stop", "length"} and res.text
    assert 0 < res.n_completion_tokens <= 640
    assert b.generate([req]) == [res]
