"""Inference backends and the backend factory.

Importing this package never imports torch, transformers or vllm: every concrete backend is imported
lazily by :func:`make_backend`.

``kind: auto`` resolves to ``vllm`` when vllm is importable and CUDA is available, else ``hf`` when CUDA is
available, else raises (CPU-only machines must choose ``mock`` or ``hf`` with ``device: cpu`` explicitly).
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING

from driftlab.backends.base import Backend, GenRequest, GenResult

if TYPE_CHECKING:  # pragma: no cover
    from driftlab.config import ExperimentConfig

__all__ = [
    "AUTO_HELP",
    "KINDS",
    "Backend",
    "GenRequest",
    "GenResult",
    "cuda_available",
    "make_backend",
    "module_available",
    "resolve_kind",
]

KINDS: tuple[str, ...] = ("mock", "hf", "vllm", "openai_compat")

AUTO_HELP = (
    "backend.kind 'auto' found no CUDA GPU (torch.cuda.is_available() is False or torch is not installed). "
    "Choose a backend explicitly: 'mock' for synthetic demos and tests (backend.kind=mock); "
    "'hf' with backend.hf.device=cpu for a slow CPU smoke test (see configs/smoke_hf_cpu.yaml); "
    "'openai_compat' with backend.openai_compat.base_url pointing at a running OpenAI-compatible server; "
    "or run on a GPU runtime (e.g. Google Colab: Runtime > Change runtime type > T4/A100 GPU), where "
    "'auto' picks vllm if installed and hf otherwise."
)


def module_available(name: str) -> bool:
    """True if ``name`` can be imported (checks ``sys.modules`` first, without importing it)."""
    if name in sys.modules:
        return sys.modules[name] is not None
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def cuda_available() -> bool:
    """``torch.cuda.is_available()``, or False when torch is missing or CUDA initialisation fails."""
    if not module_available("torch"):
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # broken CUDA driver / torch build
        return False


def resolve_kind(kind: str) -> str:
    """Concrete backend kind for a configured ``backend.kind`` (resolves ``auto``)."""
    if kind in KINDS:
        return kind
    if kind != "auto":
        raise ValueError(f"unknown backend kind {kind!r}; expected 'auto' or one of {KINDS}")
    if cuda_available():
        return "vllm" if module_available("vllm") else "hf"
    raise RuntimeError(AUTO_HELP)


def make_backend(cfg: ExperimentConfig, answer_key: Mapping[str, str] | None = None) -> Backend:
    """Construct the backend selected by ``cfg.backend.kind`` (``answer_key`` is used by the mock only)."""
    kind = resolve_kind(cfg.backend.kind)
    model_id, revision = cfg.model.id, cfg.model.revision
    if kind == "mock":
        from driftlab.backends.mock import MockBackend

        return MockBackend(model_id, revision, cfg.backend.mock, answer_key=answer_key)
    if kind == "hf":
        from driftlab.backends.hf import HFBackend

        return HFBackend(model_id, revision, cfg.backend.hf, dtype=cfg.model.dtype)
    if kind == "vllm":
        from driftlab.backends.vllm_backend import VLLMBackend

        return VLLMBackend(model_id, revision, cfg.backend.vllm, dtype=cfg.model.dtype)
    from driftlab.backends.openai_compat import OpenAICompatBackend

    return OpenAICompatBackend(model_id, revision, cfg.backend.openai_compat)
