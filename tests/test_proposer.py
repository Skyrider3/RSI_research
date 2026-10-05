"""Proposer: meta-prompt construction, leakage guard, parsing, validation and fallback edits."""

from __future__ import annotations

import dataclasses
import re

import pytest

from driftlab import keys
from driftlab.config import ExperimentConfig, ProposerSection
from driftlab.data import LeakageError, load_dev, load_eval
from driftlab.environments import Decoding
from driftlab.prompting import PROPOSER_SYSTEM, extract_current_prompt, load_prompt_file
from driftlab.proposer import (
    VIOLATIONS,
    DevError,
    build_meta_prompt,
    fallback_candidate,
    leaked_golds,
    load_fallback_edits,
    normalize_prompt,
    parse_candidate,
    proposer_request,
    sample_errors,
    shown_golds,
    truncate_response,
    validate_candidate,
)

INITIAL = load_prompt_file("initial_v1.txt")
PCFG = ProposerSection()
GOOD = (
    "Solve the problem step by step. Check each calculation carefully and put the final numeric answer "
    "within \\boxed{}."
)


@pytest.fixture(scope="module")
def cfg() -> ExperimentConfig:
    return ExperimentConfig()


@pytest.fixture(scope="module")
def dev_errors(cfg: ExperimentConfig) -> list[DevError]:
    dev = load_dev(cfg)
    # long fake responses so truncation is exercised; responses embed the item index for traceability
    return [
        DevError(it.idx, it.question, f"[resp {it.idx}] " + "work " * 300 + f"\\boxed{{{it.idx}}}", it.gold)
        for it in dev
        if it.idx % 3 == 0
    ]


# --------------------------------------------------------------------------- leakage guard


def test_dev_error_rejects_eval_split() -> None:
    DevError(1, "q", "r", "5")  # default split is "train"
    with pytest.raises(LeakageError):
        DevError(1, "q", "r", "5", split="test")
    with pytest.raises(LeakageError):
        DevError(1, "q", "r", "5", split="validation")


def test_from_item_carries_split_so_eval_items_are_refused(cfg: ExperimentConfig) -> None:
    """Regression: DevError(idx, question, response, gold) defaults to split='train', so a caller that builds
    errors from EVAL items without passing the split would slip test questions into the meta-prompt.
    DevError.from_item propagates the item's split and the guard fires."""
    dev_item, eval_item = load_dev(cfg)[0], load_eval(cfg)[0]
    e = DevError.from_item(dev_item, "resp")
    assert (e.idx, e.question, e.gold, e.split, e.response) == (
        dev_item.idx,
        dev_item.question,
        dev_item.gold,
        "train",
        "resp",
    )
    with pytest.raises(LeakageError):
        DevError.from_item(eval_item, "resp")


def test_tampered_dev_error_is_refused() -> None:
    e = DevError(1, "q", "r", "5")
    object.__setattr__(e, "split", "test")
    with pytest.raises(LeakageError):
        build_meta_prompt(INITIAL, [e], PCFG)
    with pytest.raises(LeakageError):
        sample_errors([e], 5, 0, 1)


def test_non_dev_error_objects_are_refused() -> None:
    @dataclasses.dataclass
    class Fake:
        idx: int = 0
        question: str = "q"
        response: str = "r"
        gold: str = "1"
        split: str = "train"

    with pytest.raises(TypeError):
        build_meta_prompt(INITIAL, [Fake()], PCFG)  # type: ignore[list-item]


# --------------------------------------------------------------------------- meta-prompt


def test_meta_prompt_contents(cfg: ExperimentConfig, dev_errors: list[DevError]) -> None:
    shown = sample_errors(dev_errors, PCFG.n_errors, seed=0, round_=1)
    assert len(shown) == 5
    meta = build_meta_prompt(INITIAL, shown, PCFG)
    assert f"<current_prompt>\n{INITIAL}\n</current_prompt>" in meta
    assert extract_current_prompt(meta) == INITIAL
    for i, e in enumerate(shown, start=1):
        assert f"### Problem {i}\n{e.question}\n" in meta
        assert truncate_response(e.response, 300, 300) in meta
        assert f"### Correct final answer\n{e.gold}" in meta
    assert "### Problem 6" not in meta
    assert "Below are 5 problems" in meta
    assert "<prompt> and </prompt>" in meta
    assert not re.search(r"\{(current_prompt|n_errors|error_blocks|i|question|response|gold)\}", meta)
    # the full (untruncated) responses are never shown
    assert all(e.response not in meta for e in shown)


def test_meta_prompt_contains_no_eval_text(cfg: ExperimentConfig, dev_errors: list[DevError]) -> None:
    eval_questions = load_eval(cfg).questions()
    assert len(eval_questions) == 200
    for round_ in range(1, 12):
        for seed in (0, 1, 2):
            meta = build_meta_prompt(INITIAL, sample_errors(dev_errors, 5, seed, round_), PCFG)
            assert not any(q in meta for q in eval_questions)


def test_meta_prompt_is_deterministic(dev_errors: list[DevError]) -> None:
    shown = sample_errors(dev_errors, 5, 1, 3)
    assert build_meta_prompt(INITIAL, shown, PCFG) == build_meta_prompt(INITIAL, list(shown), PCFG)


def test_meta_prompt_keeps_braces_in_values() -> None:
    e = DevError(3, "What is {x} + 1?", "\\boxed{7} {response}", "8")
    meta = build_meta_prompt("Use \\boxed{} and {curly}.", [e], PCFG)
    assert "What is {x} + 1?" in meta and "\\boxed{7} {response}" in meta
    assert "Use \\boxed{} and {curly}." in meta


def test_proposer_request() -> None:
    dec = Decoding(id="proposer", temperature=0.7)
    req = proposer_request("META", dec, keys.proposer_seed(0, 1, 0))
    assert req.system == PROPOSER_SYSTEM and req.user == "META"
    assert req.decoding == dec and req.seed == keys.proposer_seed(0, 1, 0) and req.nonce is None


# --------------------------------------------------------------------------- error sampling


def test_sample_errors_deterministic_and_seeded(dev_errors: list[DevError]) -> None:
    a = sample_errors(dev_errors, 5, 0, 1)
    assert a == sample_errors(list(reversed(dev_errors)), 5, 0, 1)  # input order irrelevant
    assert len({e.idx for e in a}) == 5
    assert {e.idx for e in a} <= {e.idx for e in dev_errors}
    draws = {tuple(e.idx for e in sample_errors(dev_errors, 5, s, r)) for s in range(3) for r in range(1, 6)}
    assert len(draws) > 10


def test_sample_errors_matches_spec(dev_errors: list[DevError]) -> None:
    import random

    ordered = sorted(dev_errors, key=lambda e: e.idx)
    want = random.Random(keys.rng_seed(2, "errors", 7)).sample(ordered, 5)
    assert sample_errors(dev_errors, 5, 2, 7) == want


def test_sample_errors_fewer_than_k(dev_errors: list[DevError]) -> None:
    few = [dev_errors[4], dev_errors[1], dev_errors[2]]
    assert [e.idx for e in sample_errors(few, 5, 0, 1)] == sorted(e.idx for e in few)
    assert sample_errors([], 5, 0, 1) == []
    assert sample_errors(dev_errors, 0, 0, 1) == []


# --------------------------------------------------------------------------- truncation


def test_truncate_response() -> None:
    assert truncate_response("short", 300, 300) == "short"
    text = "".join(chr(ord("a") + i % 26) for i in range(1000))
    out = truncate_response(text, 300, 300)
    assert out == text[:300] + " … " + text[-300:]
    assert len(out) == 603
    assert truncate_response(text, 10, 0) == text[:10] + " … "
    exact = "x" * 603
    assert truncate_response(exact, 300, 300) == exact  # truncation never lengthens
    with pytest.raises(ValueError):
        truncate_response(text, -1, 3)


# --------------------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("Here:\n<prompt>\nBe careful.\n</prompt>", "Be careful."),
        ("<prompt>first</prompt> then <prompt> second </prompt>", "second"),
        ("Use <prompt> and </prompt> tags.\n<prompt>Real one</prompt>", "Real one"),
        ("<PROMPT>Upper</Prompt>", "Upper"),
        ("<prompt>\n<prompt>\nnested\n</prompt>\n</prompt>", None),
        ("<prompt>   \n </prompt>", None),
        ("no tags at all", None),
        ("<prompt>unterminated", None),
        ("", None),
        ("<prompt>A</prompt> trailing <prompt>B", "A"),
    ],
)
def test_parse_candidate(text: str, want: str | None) -> None:
    assert parse_candidate(text) == want


# --------------------------------------------------------------------------- validation


def test_valid_candidate_has_no_violations() -> None:
    assert validate_candidate(GOOD, [INITIAL], ["72", "1234"], PCFG) == []


def test_missing() -> None:
    assert validate_candidate(None, [INITIAL], [], PCFG) == ["missing"]


def test_too_short_and_too_long() -> None:
    assert "too_short" in validate_candidate("Use \\boxed{}.", [], [], PCFG)
    long = GOOD + " Keep going." * 200
    assert len(long) > PCFG.max_chars
    assert validate_candidate(long, [], [], PCFG) == ["too_long"]
    relaxed = PCFG.model_copy(update={"min_chars": 1})
    assert validate_candidate("Use \\boxed{}.", [], [], relaxed) == []


def test_no_format_instruction() -> None:
    cand = "Solve the problem step by step and check each calculation carefully before answering."
    assert validate_candidate(cand, [], [], PCFG) == ["no_format_instruction"]
    lax = PCFG.model_copy(update={"require_format_instruction": False})
    assert validate_candidate(cand, [], [], lax) == []


def test_duplicate_detection_is_normalized() -> None:
    variant = "  " + GOOD.replace("Solve the problem", "SOLVE The Problem").replace(" ", "   \n") + "  "
    assert validate_candidate(variant, [INITIAL, GOOD], [], PCFG) == ["duplicate"]
    assert normalize_prompt("Hello,   World!!") == normalize_prompt("hello, world") == "hello, world"
    assert normalize_prompt("...Hi there.") == "hi there"


@pytest.mark.parametrize(
    ("cand_extra", "golds", "leaks"),
    [
        ("For example 1234 is a typical answer.", ["1234"], True),
        ("Totals like 1,234 happen.", ["1234"], True),
        ("Write 1234.0 if needed.", ["1234"], True),
        ("Write -150 when negative.", ["-150"], True),
        ("A value like 12345 is different.", ["1234"], False),
        ("Pi is about 3.1234.", ["1234"], False),
        ("Answers like 72 are small.", ["72"], False),  # < 3 digits: never a leak
        ("Use 2.25 as an example.", ["2.25"], True),
        ("Nothing numeric here.", ["999"], False),
    ],
)
def test_leaks_gold(cand_extra: str, golds: list[str], leaks: bool) -> None:
    cand = f"{GOOD} {cand_extra}"
    assert ("leaks_gold" in validate_candidate(cand, [], golds, PCFG)) is leaks
    assert bool(leaked_golds(cand, golds)) is leaks


def test_contains_tags() -> None:
    for tag in ("<prompt>", "</prompt>", "<current_prompt>", "</CURRENT_PROMPT>"):
        cand = f"{GOOD} {tag}"
        assert validate_candidate(cand, [], [], PCFG) == ["contains_tags"], tag


def test_violation_order_and_vocabulary() -> None:
    bad = "Copy 4567 <prompt>"
    out = validate_candidate(bad, ["copy 4567 <prompt>"], ["4567"], PCFG)
    assert out == ["too_short", "no_format_instruction", "duplicate", "leaks_gold", "contains_tags"]
    assert set(out) <= set(VIOLATIONS)
    assert out == [v for v in VIOLATIONS if v in out]


def test_shown_golds(dev_errors: list[DevError]) -> None:
    shown = dev_errors[:3]
    assert shown_golds(shown) == [e.gold for e in shown]


# --------------------------------------------------------------------------- fallback


def test_fallback_is_deterministic_and_appends_one_sentence() -> None:
    edits = load_fallback_edits()
    assert len(edits) == 8
    a = fallback_candidate(INITIAL, 0, 3, [INITIAL])
    assert a == fallback_candidate(INITIAL, 0, 3, [INITIAL])
    head, _, added = a.rpartition("\n")
    assert head == INITIAL and added in edits
    outs = {fallback_candidate(INITIAL, s, r, [INITIAL]) for s in range(3) for r in range(1, 12)}
    assert len(outs) > 1  # the seeded choice varies


def test_fallback_choice_matches_spec_formula() -> None:
    """The fallback sentence is Random(rng_seed(seed, "fallback", round)).choice over the eligible edits in
    file order (pinned: a resumed run must re-derive the same fallback)."""
    import random

    edits = load_fallback_edits()
    for seed, round_ in [(0, 1), (1, 4), (2, 11)]:
        options = [f"{INITIAL}\n{e}" for e in edits]
        want = random.Random(keys.rng_seed(seed, "fallback", round_)).choice(options)
        assert fallback_candidate(INITIAL, seed, round_, [INITIAL]) == want


def test_fallback_skips_present_sentences_and_duplicates() -> None:
    edits = load_fallback_edits()
    inc = INITIAL + "\n" + "\n".join(edits[:6])
    prior = [INITIAL, inc, f"{inc}\n{edits[6]}"]  # edits[6] would recreate a prior prompt
    for r in range(1, 12):
        out = fallback_candidate(inc, 1, r, prior)
        assert out == f"{inc}\n{edits[7]}"
        assert validate_candidate(out, prior, [], PCFG.model_copy(update={"max_chars": 5000})) == []


def test_fallback_when_all_used() -> None:
    edits = load_fallback_edits()
    inc = INITIAL + "\n" + "\n".join(edits)
    assert fallback_candidate(inc, 0, 5, [inc]) == f"{inc}\n(Revision 5.)"
    taken = f"{inc}\n(Revision 5.)"
    out = fallback_candidate(inc, 0, 5, [inc, taken])
    assert out != taken and normalize_prompt(out) != normalize_prompt(taken)


# --------------------------------------------------------------------------- mock integration


def test_mock_proposer_round_trip(cfg: ExperimentConfig, dev_errors: list[DevError]) -> None:
    mock = pytest.importorskip("driftlab.backends.mock")
    backend = mock.MockBackend(cfg.model.id, cfg.model.revision)
    shown = sample_errors(dev_errors, 5, 0, 1)
    meta = build_meta_prompt(INITIAL, shown, PCFG)
    reqs = [proposer_request(meta, cfg.proposer_decoding(), keys.proposer_seed(0, 1, a)) for a in range(20)]
    results = backend.generate(reqs)
    outcomes = []
    for req, res in zip(reqs, results, strict=True):
        assert backend.explain(req)["incumbent_found"]
        cand = parse_candidate(res.text)
        outcomes.append(validate_candidate(cand, [INITIAL], shown_golds(shown), PCFG))
    assert sum(not v for v in outcomes) >= 10  # most mock proposals are valid
