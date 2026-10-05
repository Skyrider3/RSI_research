"""Backend protocol shared by every inference engine (mock, HF transformers, vLLM, OpenAI-compatible)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from driftlab.environments import Decoding


@dataclass(frozen=True)
class GenRequest:
    """One chat completion: a system prompt (the prompt under optimization) and a user turn."""

    system: str
    user: str
    decoding: Decoding
    seed: int | None = None  # per-request sampling seed (ignored for greedy decoding)
    nonce: str | None = None  # forces a physical regeneration; never sent to the model


@dataclass
class GenResult:
    text: str
    finish_reason: str  # "stop" | "length" | "error"
    n_prompt_tokens: int = 0
    n_completion_tokens: int = 0
    latency_ms: float = 0.0


class Backend(ABC):
    """Inference engine.

    Implementations MUST honour every field of ``Decoding`` explicitly (temperature, top_p, top_k,
    repetition_penalty, max_new_tokens) and must never fall back to a model's own
    ``generation_config.json`` defaults (Qwen2.5 ships do_sample=True, T=0.7, top_p=0.8, top_k=20,
    repetition_penalty=1.05/1.1, which would silently turn "greedy" into sampling).
    """

    kind: str = "base"
    synthetic: bool = False  # True only for the mock backend; propagated to runs.synthetic

    def __init__(self, model_id: str, model_revision: str) -> None:
        self.model_id = model_id
        self.model_revision = model_revision

    @abstractmethod
    def engine_info(self) -> dict:
        """Everything about the engine that could change outputs (kind, versions, dtype, GPU class, ...)."""

    @abstractmethod
    def render(self, system: str, user: str) -> str:
        """Render the chat template to the exact prompt string fed to the model (hashed into gen_key)."""

    @abstractmethod
    def generate(self, reqs: Sequence[GenRequest]) -> list[GenResult]:
        """Generate one result per request, in order."""

    def close(self) -> None:  # pragma: no cover - optional
        return None
