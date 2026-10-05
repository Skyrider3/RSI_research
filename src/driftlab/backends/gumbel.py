"""Per-row Gumbel-max sampling for Hugging Face ``generate`` (exact, batch-composition independent).

``argmax(logits / T + G)`` with ``G`` i.i.d. standard Gumbel noise is an *exact* sample from
``softmax(logits / T)`` (the Gumbel-max trick). Drawing the noise of every row from that row's own
``torch.Generator`` seeded with the request seed makes each row's sample depend only on its own seed and
logits, never on which other requests share the batch or where the row sits in it. Every row advances its
generator exactly once per decoding step (finished rows included), so row ``i`` at step ``s`` always
consumes the ``s``-th draw of its own stream.

The processor is used with ``do_sample=False``: ``generate`` then takes the argmax of the processed
scores, which *is* the sample. Optional top-k / top-p truncation (``top_k=0`` / ``top_p=1.0`` mean
disabled) masks tokens to ``-inf`` before the noise is added, giving an exact sample from the
renormalised truncated distribution.

This module imports without torch/transformers; ``PerRowGumbelProcessor`` (resolved lazily) and
:func:`make_gumbel_processor` raise a clear ``ImportError`` only when used without them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

SAMPLER_VERSION = "gumbel-per-row/v1"  # recorded in the HF engine fingerprint

_SEED_MASK_64 = 0xFFFF_FFFF_FFFF_FFFF
_CLASS: Any = None  # the built processor class (cached after the first successful build)


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as e:  # pragma: no cover - exercised via sys.modules patching in tests
        raise ImportError(
            "PerRowGumbelProcessor needs torch and transformers (pip install 'driftlab[hf]'); "
            f"import failed: {e}"
        ) from e
    return torch, transformers


def _filter_top_k(torch: Any, scores: Any, top_k: int) -> Any:
    if top_k <= 0 or top_k >= scores.shape[-1]:
        return scores
    kth = torch.topk(scores, top_k, dim=-1).values[..., -1:]
    return scores.masked_fill(scores < kth, float("-inf"))


def _filter_top_p(torch: Any, scores: Any, top_p: float) -> Any:
    """Nucleus filter with the same semantics as transformers' ``TopPLogitsWarper`` (keeps >= 1 token)."""
    if top_p >= 1.0:
        return scores
    sorted_scores, sorted_idx = torch.sort(scores, descending=False, stable=True, dim=-1)
    cum = sorted_scores.softmax(dim=-1).cumsum(dim=-1)
    remove_sorted = cum <= (1.0 - top_p)
    remove_sorted[..., -1] = False  # always keep the most likely token
    remove = remove_sorted.scatter(-1, sorted_idx, remove_sorted)
    return scores.masked_fill(remove, float("-inf"))


def _build_class() -> Any:
    torch, transformers = _require_torch()

    class PerRowGumbelProcessor(transformers.LogitsProcessor):
        """Logits processor turning greedy ``generate`` into exact per-row seeded sampling.

        Args:
            seeds: one seed per batch row (row order = the order of ``input_ids`` rows).
            temperature: sampling temperature (> 0).
            top_p: nucleus mass; ``1.0`` disables.
            top_k: keep the k most likely tokens; ``0`` disables.
            device: device of the per-row generators (use the model's device so the noise is drawn there).
        """

        def __init__(
            self,
            seeds: Sequence[int],
            temperature: float,
            top_p: float = 1.0,
            top_k: int = 0,
            device: Any = "cpu",
        ) -> None:
            if not seeds:
                raise ValueError("PerRowGumbelProcessor needs at least one seed")
            if not temperature > 0:
                raise ValueError(f"temperature must be > 0 for sampling, got {temperature!r}")
            if not 0.0 < top_p <= 1.0:
                raise ValueError(f"top_p must be in (0, 1], got {top_p!r}")
            if top_k < 0:
                raise ValueError(f"top_k must be >= 0 (0 = disabled), got {top_k!r}")
            self.seeds = [int(s) for s in seeds]
            self.temperature = float(temperature)
            self.top_p = float(top_p)
            self.top_k = int(top_k)
            self.device = torch.device(device)
            self.generators = []
            for s in self.seeds:
                g = torch.Generator(device=self.device)
                g.manual_seed(s & _SEED_MASK_64)
                self.generators.append(g)
            self.steps = 0
            finfo = torch.finfo(torch.float32)
            self._u_min = finfo.tiny
            self._u_max = 1.0 - finfo.eps

        def noise(self, vocab_size: int) -> Any:
            """One step of standard Gumbel noise, shape ``[rows, vocab_size]`` (float32)."""
            u = torch.stack(
                [
                    torch.rand(vocab_size, generator=g, device=self.device, dtype=torch.float32)
                    for g in self.generators
                ]
            )
            u = u.clamp_(min=self._u_min, max=self._u_max)
            return -torch.log(-torch.log(u))

        def __call__(self, input_ids: Any, scores: Any) -> Any:
            if scores.shape[0] != len(self.generators):
                raise ValueError(
                    f"PerRowGumbelProcessor was built for {len(self.generators)} rows, "
                    f"got scores for {scores.shape[0]}"
                )
            s = scores.float() / self.temperature
            s = _filter_top_k(torch, s, self.top_k)
            s = _filter_top_p(torch, s, self.top_p)
            g = self.noise(s.shape[-1]).to(s.device)
            self.steps += 1
            return s + g

    PerRowGumbelProcessor.__module__ = __name__
    return PerRowGumbelProcessor


def get_processor_class() -> Any:
    """The ``PerRowGumbelProcessor`` class (builds it on first use; needs torch + transformers)."""
    global _CLASS
    if _CLASS is None:
        _CLASS = _build_class()
    return _CLASS


def make_gumbel_processor(
    seeds: Sequence[int], temperature: float, top_p: float = 1.0, top_k: int = 0, device: Any = "cpu"
) -> Any:
    """Construct a ``PerRowGumbelProcessor`` (raises ``ImportError`` if torch/transformers are missing)."""
    return get_processor_class()(seeds, temperature, top_p=top_p, top_k=top_k, device=device)


def __getattr__(
    name: str,
) -> Any:  # PEP 562: lazy ``from driftlab.backends.gumbel import PerRowGumbelProcessor``
    if name == "PerRowGumbelProcessor":
        return get_processor_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
