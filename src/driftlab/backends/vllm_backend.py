"""vLLM offline backend (preferred on Colab GPUs).

* ``LLM(..., generation_config="vllm")`` so vLLM never applies the checkpoint's ``generation_config.json``
  sampling defaults (older vLLM versions without that argument are detected and fall back to plain
  construction; explicit ``SamplingParams`` are passed for every request either way).
* ``enable_prefix_caching`` defaults to False: with prefix caching, physical reruns would share KV state
  with earlier generations of the same prompt.
* Every request gets explicit ``SamplingParams(n=1, temperature, top_p, top_k, repetition_penalty,
  max_tokens, seed, stop_token_ids)``; the seed is sent only for sampling. "top_k disabled" is spelled the
  way the installed vLLM spells it (historically -1, newer versions use 0).
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any

from driftlab.backends.base import Backend, GenRequest, GenResult
from driftlab.backends.hf import (
    RenderCache,
    effective_seed,
    format_capability,
    package_version,
    render_chat,
    resolve_dtype_name,
    stop_token_ids,
)
from driftlab.config import VLLMSection

LLM_SEED = 0
# Environment overrides that switch vLLM's engine / attention kernels without any version change.
ENGINE_ENV_VARS: tuple[str, ...] = ("VLLM_USE_V1", "VLLM_ATTENTION_BACKEND")


def _device_info() -> dict[str, Any] | None:
    """``{"name", "capability"}`` of CUDA device 0, or None without torch / CUDA."""
    try:
        import torch
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return {
            "name": str(torch.cuda.get_device_name(0)),
            "capability": tuple(torch.cuda.get_device_capability(0)),
        }
    except Exception:
        return None


def _torch_version() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    return str(torch.__version__)


def disabled_top_k(sampling_params_cls: Any) -> int:
    """The installed vLLM's value for "top_k disabled": its own default when that is -1 or 0, else -1 if
    accepted, else 0."""
    try:
        default = sampling_params_cls().top_k
        if default in (-1, 0):
            return int(default)
    except Exception:
        pass
    try:
        sampling_params_cls(top_k=-1)
        return -1
    except Exception:
        return 0


class VLLMBackend(Backend):
    """vLLM ``LLM`` engine with explicit per-request sampling parameters.

    Args:
        model_id, model_revision: Hub id and pinned commit (used for weights and tokenizer).
        settings: ``cfg.backend.vllm``.
        dtype: ``cfg.model.dtype`` (``auto`` -> bf16 on sm >= 80, fp16 on older GPUs).
        llm: optional pre-built ``vllm.LLM`` (notebooks that already hold one). Its construction is the
            caller's responsibility, so ``engine_info()["generation_config"]`` reports ``external-llm``.
    """

    kind = "vllm"

    def __init__(
        self,
        model_id: str,
        model_revision: str,
        settings: VLLMSection | None = None,
        *,
        dtype: str = "auto",
        llm: Any = None,
    ) -> None:
        super().__init__(model_id, model_revision)
        try:
            import vllm
        except ImportError as e:
            raise ImportError(
                "the 'vllm' backend needs vllm (pip install vllm; on Colab restart the runtime afterwards), "
                f"or use backend.kind=hf. Import failed: {e}"
            ) from e
        self._vllm = vllm
        self._sampling_params_cls = vllm.SamplingParams
        self.settings = settings or VLLMSection()
        dev = _device_info()
        self.gpu_class = dev["name"] if dev else "cpu"
        self.capability: tuple[int, int] | None = dev["capability"] if dev else None
        self.dtype_name = resolve_dtype_name(dtype, "cuda" if dev else "cpu", self.capability)
        self.generation_config_mode = "vllm"
        self.llm_kwargs: dict[str, Any] = {
            "model": model_id,
            "revision": model_revision,
            "tokenizer_revision": model_revision,
            "dtype": self.dtype_name,
            "seed": LLM_SEED,
            "gpu_memory_utilization": float(self.settings.gpu_memory_utilization),
            "max_model_len": int(self.settings.max_model_len),
            "enable_prefix_caching": bool(self.settings.enable_prefix_caching),
            "max_num_seqs": int(self.settings.max_num_seqs),
            "enforce_eager": bool(self.settings.enforce_eager),
            "generation_config": "vllm",
        }
        if llm is not None:
            self.generation_config_mode = "external-llm"  # built by the caller; we cannot vouch for it
            self.llm = llm
        else:
            self.llm = self._build_llm()
        self.tokenizer = self.llm.get_tokenizer()
        self.stop_ids = stop_token_ids(self.tokenizer)
        self.top_k_disabled = disabled_top_k(self._sampling_params_cls)
        self._renders = RenderCache()

    def _build_llm(self) -> Any:
        try:
            return self._vllm.LLM(**self.llm_kwargs)
        except TypeError as e:
            if "generation_config" not in str(e):
                raise
            # Old vLLM without the argument: it does not read generation_config.json for offline
            # SamplingParams anyway, and every request carries explicit parameters.
            self.llm_kwargs.pop("generation_config")
            self.generation_config_mode = "unsupported(explicit-params)"
            return self._vllm.LLM(**self.llm_kwargs)

    def _effective_dtype(self) -> str:
        try:
            return str(self.llm.llm_engine.model_config.dtype).removeprefix("torch.")
        except Exception:
            return self.dtype_name

    # ------------------------------------------------------------------ protocol
    def engine_info(self) -> dict:
        return {
            "kind": "vllm",
            "vllm": str(getattr(self._vllm, "__version__", "unknown")),
            "torch": _torch_version(),
            "transformers": package_version("transformers"),  # chat template + tokenizer code
            "tokenizers": package_version("tokenizers"),
            "env": {k: os.environ.get(k) for k in ENGINE_ENV_VARS},
            "dtype": self._effective_dtype(),
            "gpu_class": self.gpu_class,
            "capability": format_capability(self.capability),
            "prefix_caching": bool(self.settings.enable_prefix_caching),
            "enforce_eager": bool(self.settings.enforce_eager),
            "max_num_seqs": int(self.settings.max_num_seqs),
            "max_model_len": int(self.settings.max_model_len),
            "gpu_memory_utilization": float(self.settings.gpu_memory_utilization),
            "generation_config": self.generation_config_mode,
            "seed": LLM_SEED,
            "top_k_disabled": self.top_k_disabled,
            "stop_token_ids": list(self.stop_ids),
        }

    def render(self, system: str, user: str) -> str:
        key = (system, user)
        out = self._renders.get(key)
        if out is None:
            out = render_chat(self.tokenizer, system, user)
            self._renders.put(key, out)
        return out

    def sampling_params(self, req: GenRequest, rendered: str) -> Any:
        """Explicit ``SamplingParams`` for one request (seed only when sampling)."""
        d = req.decoding
        return self._sampling_params_cls(
            n=1,
            temperature=float(d.temperature),
            top_p=float(d.top_p),
            top_k=int(d.top_k) if d.top_k > 0 else self.top_k_disabled,
            repetition_penalty=float(d.repetition_penalty),
            max_tokens=int(d.max_new_tokens),
            seed=effective_seed(req, rendered),
            stop_token_ids=list(self.stop_ids),
        )

    def generate(self, reqs: Sequence[GenRequest]) -> list[GenResult]:
        if not reqs:
            return []
        prompts = [self.render(r.system, r.user) for r in reqs]
        params = [self.sampling_params(r, p) for r, p in zip(reqs, prompts, strict=True)]
        t0 = time.perf_counter()
        outs = self.llm.generate(prompts, sampling_params=params, use_tqdm=False)
        latency_ms = (time.perf_counter() - t0) * 1000.0 / len(reqs)
        if len(outs) != len(reqs):
            raise RuntimeError(f"vLLM returned {len(outs)} outputs for {len(reqs)} prompts")
        results = []
        for out in outs:
            o = out.outputs[0]
            fr = o.finish_reason
            finish = "length" if fr == "length" else "stop" if fr == "stop" else "error"
            text = o.text if finish != "error" else f"vLLM finish_reason={fr!r}: {o.text[:200]}"
            results.append(
                GenResult(
                    text=text,
                    finish_reason=finish,
                    n_prompt_tokens=len(getattr(out, "prompt_token_ids", None) or []),
                    n_completion_tokens=len(o.token_ids or []),
                    latency_ms=latency_ms,
                )
            )
        return results

    def close(self) -> None:
        self.llm = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
