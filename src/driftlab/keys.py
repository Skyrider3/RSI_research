"""Content-addressed keys and deterministic seed derivation.

Design rules (see docs/METHODS.md, "Cache semantics"):

* A *generation* is identified by ``gen_key``: a hash of everything that determines the model output
  (engine, model revision, the fully rendered chat prompt, explicit decoding params, the sampling seed
  and an optional *nonce*). Greedy keys ignore the seed, so identical greedy requests share one output.
* A *physical rerun* adds ``nonce="rerun:<draw>"`` so it can never be served from the cache. Without a
  nonce, a greedy "rerun" resolves to the stored generation and is identical *by construction*; the
  analyses detect and flag that case instead of reporting it as a measured zero.
* Sampling seeds are derived from the logical *slot* (not the prompt text), so two slots whose prompt
  texts happen to coincide still get independent samples.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from driftlab.environments import Decoding

SEED_MASK = 0x7FFFFFFF  # 31-bit seeds are accepted by every backend


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_json(obj: object) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def gen_key(
    *,
    rendered_sha: str,
    decoding: Decoding,
    seed: int | None,
    nonce: str | None,
    model_id: str,
    model_revision: str,
    engine_fp: str,
) -> str:
    """Key of one physical generation. ``seed`` is ignored for greedy decoding."""
    return sha256_json(
        {
            "v": 1,
            "engine": engine_fp,
            "model": model_id,
            "revision": model_revision,
            "prompt": rendered_sha,
            "decoding": decoding.params(),
            "seed": None if decoding.is_greedy else seed,
            "nonce": nonce,
        }
    )


def _derive(*parts: object) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") & SEED_MASK


def sample_seed(
    run_seed: int,
    purpose: str,
    decoding_id: str,
    slot: int,
    draw_kind: str,
    draw: int,
    split: str,
    item_idx: int,
) -> int:
    """Per-request sampling seed for evaluation generations."""
    return _derive("sample", run_seed, purpose, decoding_id, slot, draw_kind, draw, split, item_idx)


def proposer_seed(run_seed: int, round_: int, attempt: int) -> int:
    return _derive("proposer", run_seed, round_, attempt)


def rng_seed(*parts: object) -> int:
    """Generic deterministic seed for numpy / random generators (error sampling, bootstrap, ...)."""
    return _derive("rng", *parts)


def engine_fingerprint(info: Mapping[str, object]) -> str:
    """Fingerprint of the inference engine (backend kind/version, torch, dtype, GPU class, attention, ...)."""
    return sha256_json(dict(info))[:16]
