"""Key semantics: content-addressed generation keys and deterministic seed derivation."""

from __future__ import annotations

import pytest

from driftlab.environments import Decoding
from driftlab.keys import (
    SEED_MASK,
    engine_fingerprint,
    gen_key,
    proposer_seed,
    rng_seed,
    sample_seed,
    sha256_json,
    sha256_text,
)

GREEDY = Decoding(id="greedy", temperature=0.0)
T02 = Decoding(id="t02", temperature=0.2)


def _key(**over: object) -> str:
    args: dict = {
        "rendered_sha": sha256_text("system\nuser"),
        "decoding": GREEDY,
        "seed": None,
        "nonce": None,
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "model_revision": "rev1",
        "engine_fp": "eng1",
    }
    args.update(over)
    return gen_key(**args)


def test_gen_key_is_deterministic_hex():
    k = _key()
    assert k == _key()
    assert len(k) == 64 and int(k, 16) >= 0


def test_nonce_changes_key():
    assert _key(nonce="rerun:3") != _key()
    assert _key(nonce="rerun:3") != _key(nonce="rerun:4")
    assert _key(nonce="audit:0") == _key(nonce="audit:0")


def test_greedy_key_ignores_seed():
    assert _key(seed=1) == _key(seed=2) == _key(seed=None)


def test_sampling_key_depends_on_seed():
    assert _key(decoding=T02, seed=1) != _key(decoding=T02, seed=2)
    assert _key(decoding=T02, seed=1) == _key(decoding=T02, seed=1)
    assert _key(decoding=T02, seed=1) != _key(decoding=GREEDY, seed=1)


def test_decoding_params_change_key_but_not_id():
    assert _key(decoding=Decoding(id="greedy", temperature=0.0, max_new_tokens=48)) != _key()
    assert _key(decoding=Decoding(id="greedy", temperature=0.0, top_p=0.9)) != _key()
    # the id is a label, the explicit parameters define the generation
    assert _key(decoding=Decoding(id="other_name", temperature=0.0)) == _key()


@pytest.mark.parametrize(
    "field,value",
    [
        ("engine_fp", "eng2"),
        ("model_revision", "rev2"),
        ("model_id", "Qwen/Qwen2.5-0.5B-Instruct"),
        ("rendered_sha", sha256_text("system\nother user")),
    ],
)
def test_environment_components_change_key(field: str, value: str):
    assert _key(**{field: value}) != _key()


def test_engine_fingerprint_order_independent_and_sensitive():
    a = engine_fingerprint({"kind": "vllm", "version": "0.6.3", "gpu": "T4"})
    b = engine_fingerprint({"gpu": "T4", "version": "0.6.3", "kind": "vllm"})
    assert a == b and len(a) == 16
    assert engine_fingerprint({"kind": "vllm", "version": "0.6.3", "gpu": "A100"}) != a


def test_sample_seed_deterministic_and_in_range():
    s = sample_seed(0, "eval", "t02", 3, "round", 5, "test", 17)
    assert s == sample_seed(0, "eval", "t02", 3, "round", 5, "test", 17)
    assert 0 <= s <= SEED_MASK


def test_sample_seed_differs_across_every_component():
    base = ("eval", "t02", 3, "round", 5, "test", 17)
    s0 = sample_seed(0, *base)
    variants = [
        sample_seed(1, *base),  # run seed
        sample_seed(0, "gt", "t02", 3, "round", 5, "test", 17),  # purpose
        sample_seed(0, "eval", "t07", 3, "round", 5, "test", 17),  # decoding
        sample_seed(0, "eval", "t02", 4, "round", 5, "test", 17),  # slot
        sample_seed(0, "eval", "t02", 3, "gt", 5, "test", 17),  # draw kind
        sample_seed(0, "eval", "t02", 3, "round", 6, "test", 17),  # draw
        sample_seed(0, "eval", "t02", 3, "round", 5, "train", 17),  # split
        sample_seed(0, "eval", "t02", 3, "round", 5, "test", 18),  # item
    ]
    assert all(v != s0 for v in variants)
    assert len(set(variants)) == len(variants)


def test_sample_seeds_are_spread_over_items():
    seeds = {sample_seed(0, "eval", "t02", 0, "round", 0, "test", n) for n in range(200)}
    assert len(seeds) == 200


def test_proposer_seed_deterministic_and_distinct():
    assert proposer_seed(0, 1, 0) == proposer_seed(0, 1, 0)
    grid = {proposer_seed(s, t, a) for s in range(3) for t in range(1, 12) for a in range(3)}
    assert len(grid) == 3 * 11 * 3
    assert all(0 <= x <= SEED_MASK for x in grid)


def test_namespaces_do_not_collide():
    # identical numeric parts in different namespaces give different seeds
    assert rng_seed(0, 1, 0) != proposer_seed(0, 1, 0)
    assert rng_seed(0, "errors", 1) == rng_seed(0, "errors", 1)
    assert rng_seed(0, "errors", 1) != rng_seed(1, "errors", 1)


def test_sha256_json_canonical():
    assert sha256_json({"a": 1, "b": [1, 2]}) == sha256_json({"b": [1, 2], "a": 1})
    assert sha256_json({"a": 1}) != sha256_json({"a": 2})
