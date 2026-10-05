"""Hugging Face transformers backend (Colab T4 fp16 / A100 bf16; CPU fp32 for smoke tests).

Decoding is always fully explicit: every call builds a fresh ``transformers.GenerationConfig`` and the
model's own ``generation_config`` (Qwen2.5 ships do_sample=True, T=0.7, top_p=0.8, top_k=20,
repetition_penalty>1) is replaced by a neutral one, because transformers fills unset fields from it
(v5: every ``None`` field; v4.50+: every field left at its global default).

* Greedy: ``do_sample=False``; argmax decoding.
* Sampling (T > 0): still ``do_sample=False``, plus :class:`~driftlab.backends.gumbel.PerRowGumbelProcessor`
  so each row is an exact sample from ``softmax(logits / T)`` (optionally top-k / top-p truncated) driven by
  its own ``torch.Generator(req.seed)``, independent of batch composition.
* Batches are planned deterministically: requests are grouped by decoding parameters and sorted by
  (decoding, rendered-length bucket descending, original index), then cut into ``batch_size`` batches.
  Batch size is part of :meth:`HFBackend.engine_info` because padding can perturb low-precision greedy
  outputs.

The module-level helpers (dtype resolution, chat rendering, stop tokens, render cache, seeds) import
without torch and are shared with the vLLM and OpenAI-compatible backends.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import re
import time
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

from driftlab.backends.base import Backend, GenRequest, GenResult
from driftlab.backends.gumbel import SAMPLER_VERSION, make_gumbel_processor
from driftlab.config import HFSection
from driftlab.environments import Decoding
from driftlab.keys import rng_seed, sha256_text

LENGTH_BUCKET = 16  # tokens per rendered-length bucket in the batch plan
BATCH_PLAN_VERSION = f"decoding/len{LENGTH_BUCKET}desc/index/v1"
STOP_TOKENS: tuple[str, ...] = ("<|im_end|>", "<|endoftext|>")
DEFAULT_PAD_TOKEN = "<|endoftext|>"
ATTN_FALLBACKS: tuple[str, ...] = ("sdpa", "eager")

_DTYPE_ALIASES = {
    "auto": "auto",
    "float32": "float32",
    "fp32": "float32",
    "float": "float32",
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
}


# --------------------------------------------------------------------------- torch-free helpers


def normalize_dtype_name(name: str) -> str:
    """Canonical dtype string (``auto`` | ``float32`` | ``float16`` | ``bfloat16``)."""
    key = str(name).strip().lower().removeprefix("torch.")
    if key not in _DTYPE_ALIASES:
        raise ValueError(f"unknown dtype {name!r}; use one of {sorted(set(_DTYPE_ALIASES))}")
    return _DTYPE_ALIASES[key]


def resolve_dtype_name(name: str, device: str, capability: tuple[int, int] | None = None) -> str:
    """``auto`` -> bfloat16 on CUDA compute capability >= 8.0, float16 on older CUDA GPUs, float32 on CPU.

    Explicit dtype strings are honoured (after alias normalisation).
    """
    canon = normalize_dtype_name(name)
    if canon != "auto":
        return canon
    if str(device).startswith("cuda"):
        if capability is None:
            raise ValueError("resolving dtype 'auto' on CUDA needs the device compute capability")
        return "bfloat16" if tuple(capability) >= (8, 0) else "float16"
    return "float32"


def format_capability(capability: tuple[int, int] | None) -> str | None:
    return None if capability is None else f"{capability[0]}.{capability[1]}"


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for p in str(version).split(".")[:3]:
        m = re.match(r"\d+", p)
        if not m:
            break
        parts.append(int(m.group()))
    return tuple(parts)


def package_version(name: str) -> str | None:
    """Installed distribution version (without importing it), or None if absent."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def transformers_dtype_kwarg(version: str) -> str:
    """Name of the ``from_pretrained`` dtype keyword: ``dtype`` (transformers >= 4.56, v5) else ``torch_dtype``."""
    return "dtype" if _version_tuple(version) >= (4, 56) else "torch_dtype"


def render_chat(tokenizer: Any, system: str, user: str) -> str:
    """Exact prompt string: the tokenizer's chat template with a generation prompt appended."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def stop_token_ids(tokenizer: Any) -> list[int]:
    """``[eos_token_id, <|im_end|>, <|endoftext|>]`` (those that exist), deduplicated, order kept."""
    ids: list[int] = []
    unk = getattr(tokenizer, "unk_token_id", None)
    candidates: list[Any] = [getattr(tokenizer, "eos_token_id", None)]
    for tok in STOP_TOKENS:
        try:
            candidates.append(tokenizer.convert_tokens_to_ids(tok))
        except (KeyError, ValueError, TypeError):
            continue
    for c in candidates:
        if isinstance(c, int) and c >= 0 and (unk is None or c != unk) and c not in ids:
            ids.append(c)
    return ids


def effective_seed(req: GenRequest, rendered: str) -> int | None:
    """Sampling seed actually used: ``req.seed``, or (unseeded sampling) a seed derived from the rendered
    prompt so that no wall-clock randomness ever enters. ``None`` for greedy requests."""
    if req.decoding.is_greedy:
        return None
    if req.seed is not None:
        return int(req.seed)
    return rng_seed("unseeded-sampling", sha256_text(rendered), req.decoding.params())


def decoding_sort_key(d: Decoding) -> tuple[float, float, int, float, int]:
    """Batching key: requests with numerically identical decoding parameters may share a batch."""
    return (
        float(d.temperature),
        float(d.top_p),
        int(d.top_k),
        float(d.repetition_penalty),
        int(d.max_new_tokens),
    )


def batch_plan(decodings: Sequence[Decoding], lengths: Sequence[int], batch_size: int) -> list[list[int]]:
    """Deterministic batches of request indices.

    Sort by (decoding parameters, rendered-length bucket descending, original index), then cut each decoding
    group into consecutive batches of at most ``batch_size``. Batches never mix decodings.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if len(decodings) != len(lengths):
        raise ValueError("decodings and lengths must have the same length")
    order = sorted(
        range(len(lengths)),
        key=lambda i: (decoding_sort_key(decodings[i]), -(int(lengths[i]) // LENGTH_BUCKET), i),
    )
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_key: tuple | None = None
    for i in order:
        k = decoding_sort_key(decodings[i])
        if cur and (k != cur_key or len(cur) >= batch_size):
            batches.append(cur)
            cur = []
        cur.append(i)
        cur_key = k
    if cur:
        batches.append(cur)
    return batches


class RenderCache:
    """Small LRU cache for ``render(system, user)`` (the engine renders every request at least twice)."""

    def __init__(self, maxsize: int = 8192) -> None:
        self.maxsize = int(maxsize)
        self._data: OrderedDict[tuple[str, str], str] = OrderedDict()

    def get(self, key: tuple[str, str]) -> str | None:
        v = self._data.get(key)
        if v is not None:
            self._data.move_to_end(key)
        return v

    def put(self, key: tuple[str, str], value: str) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)


def _import_torch_transformers() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as e:
        raise ImportError(
            f"the 'hf' backend needs torch and transformers: pip install 'driftlab[hf]' (import failed: {e})"
        ) from e
    return torch, transformers


# --------------------------------------------------------------------------- backend


class HFBackend(Backend):
    """Hugging Face transformers inference with explicit decoding and exact per-row seeded sampling.

    Args:
        model_id, model_revision: Hub id and pinned commit.
        settings: ``cfg.backend.hf`` (batch_size, attn_impl, device ``auto|cuda|cpu``).
        dtype: ``cfg.model.dtype`` (``auto`` -> bf16 on sm >= 80, fp16 on older GPUs, fp32 on CPU).
        model, tokenizer: optional pre-loaded objects (tests); the model is moved/cast to the resolved
            device/dtype and its ``generation_config`` is neutralised exactly like a freshly loaded one.
    """

    kind = "hf"

    def __init__(
        self,
        model_id: str,
        model_revision: str,
        settings: HFSection | None = None,
        *,
        dtype: str = "auto",
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        super().__init__(model_id, model_revision)
        torch, transformers = _import_torch_transformers()
        self._torch = torch
        self._transformers = transformers
        self.settings = settings or HFSection()
        self.batch_size = int(self.settings.batch_size)
        if self.batch_size < 1:
            raise ValueError("backend.hf.batch_size must be >= 1")

        self.device = self._resolve_device(self.settings.device)
        self.device_type = torch.device(self.device).type
        if self.device_type == "cuda":
            self.capability: tuple[int, int] | None = tuple(torch.cuda.get_device_capability(self.device))
            self.gpu_class = str(torch.cuda.get_device_name(self.device))
        else:
            self.capability = None
            self.gpu_class = "cpu"
        self.dtype_name = resolve_dtype_name(dtype, self.device_type, self.capability)
        self.torch_dtype = getattr(torch, self.dtype_name)

        if tokenizer is None:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, revision=model_revision)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            pad_id = tokenizer.convert_tokens_to_ids(DEFAULT_PAD_TOKEN)
            has_default = isinstance(pad_id, int) and pad_id != getattr(tokenizer, "unk_token_id", None)
            tokenizer.pad_token = DEFAULT_PAD_TOKEN if has_default else tokenizer.eos_token
        self.tokenizer = tokenizer
        self.pad_token_id = int(tokenizer.pad_token_id)
        self.stop_ids = stop_token_ids(tokenizer)
        if not self.stop_ids:
            raise ValueError(f"tokenizer of {model_id} defines no EOS / <|im_end|> / <|endoftext|> token")

        if model is None:
            model, attn = self._load_model()
        else:
            model = model.to(device=self.device, dtype=self.torch_dtype)
            attn = str(getattr(model.config, "_attn_implementation", None) or "unknown")
        model.eval()
        self.model = model
        self.attn_impl = attn
        gen_cfg = getattr(model, "generation_config", None)
        # The checkpoint's own defaults, kept for provenance only (never used for decoding).
        self.model_generation_defaults: dict = gen_cfg.to_dict() if gen_cfg is not None else {}
        self._neutralize_model_generation_config()
        params = inspect.signature(model.generate).parameters
        self._use_model_defaults_kw = "use_model_defaults" in params
        self._renders = RenderCache()

    # ------------------------------------------------------------------ setup
    def _resolve_device(self, setting: str) -> str:
        torch = self._torch
        setting = (setting or "auto").strip().lower()
        if setting == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if setting.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "backend.hf.device is 'cuda' but torch.cuda.is_available() is False "
                "(use device: cpu for smoke tests, or a GPU runtime on Colab)"
            )
        return setting

    def _load_model(self) -> tuple[Any, str]:
        transformers = self._transformers
        dtype_kw = transformers_dtype_kwarg(transformers.__version__)
        requested = self.settings.attn_impl or "sdpa"
        errors: list[str] = []
        for impl in dict.fromkeys((requested, *ATTN_FALLBACKS)):
            try:
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    revision=self.model_revision,
                    attn_implementation=impl,
                    **{dtype_kw: self.torch_dtype},
                )
            except (ValueError, ImportError) as e:  # e.g. flash_attention_2 not installed / unsupported
                errors.append(f"{impl}: {e}")
                continue
            model = model.to(self.device)
            used = getattr(model.config, "_attn_implementation", None) or impl
            return model, str(used)
        raise RuntimeError(f"could not load {self.model_id} with any attention implementation: {errors}")

    def _base_generation_kwargs(self, max_new_tokens: int, repetition_penalty: float) -> dict[str, Any]:
        return {
            "do_sample": False,
            "num_beams": 1,
            "num_return_sequences": 1,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "repetition_penalty": float(repetition_penalty),
            "max_new_tokens": int(max_new_tokens),
            "eos_token_id": list(self.stop_ids),
            "pad_token_id": self.pad_token_id,
            "use_cache": True,
        }

    def _neutralize_model_generation_config(self) -> None:
        """Replace the checkpoint's generation defaults so nothing can be inherited from them."""
        gc_cls = self._transformers.GenerationConfig
        self._neutral_config = gc_cls(
            **self._base_generation_kwargs(max_new_tokens=1, repetition_penalty=1.0)
        )
        self.model.generation_config = self._neutral_config

    def generation_config(self, decoding: Decoding) -> Any:
        """Fresh, fully explicit ``GenerationConfig`` for one decoding (sampling is done by the processor)."""
        return self._transformers.GenerationConfig(
            **self._base_generation_kwargs(decoding.max_new_tokens, decoding.repetition_penalty)
        )

    # ------------------------------------------------------------------ protocol
    def engine_info(self) -> dict:
        return {
            "kind": "hf",
            "transformers": self._transformers.__version__,
            "tokenizers": package_version("tokenizers"),  # tokenization of the rendered prompt
            "torch": self._torch.__version__,
            "dtype": self.dtype_name,
            "device": self.device_type,
            "gpu_class": self.gpu_class,
            "capability": format_capability(self.capability),
            "attn": self.attn_impl,
            "batch_size": self.batch_size,
            "batch_plan": BATCH_PLAN_VERSION,
            "padding_side": "left",
            "sampler": SAMPLER_VERSION,
            "stop_token_ids": list(self.stop_ids),
        }

    def render(self, system: str, user: str) -> str:
        key = (system, user)
        out = self._renders.get(key)
        if out is None:
            out = render_chat(self.tokenizer, system, user)
            self._renders.put(key, out)
        return out

    def plan_batches(self, reqs: Sequence[GenRequest]) -> list[list[int]]:
        """Deterministic batch plan (lists of request indices) that :meth:`generate` will execute."""
        prompts = [self.render(r.system, r.user) for r in reqs]
        lengths = [len(ids) for ids in self.tokenizer(prompts, add_special_tokens=False)["input_ids"]]
        return batch_plan([r.decoding for r in reqs], lengths, self.batch_size)

    def generate(self, reqs: Sequence[GenRequest]) -> list[GenResult]:
        if not reqs:
            return []
        prompts = [self.render(r.system, r.user) for r in reqs]
        lengths = [len(ids) for ids in self.tokenizer(prompts, add_special_tokens=False)["input_ids"]]
        results: list[GenResult | None] = [None] * len(reqs)
        for batch in batch_plan([r.decoding for r in reqs], lengths, self.batch_size):
            batch_reqs = [reqs[i] for i in batch]
            outs = self._generate_batch(batch_reqs, [prompts[i] for i in batch])
            for i, res in zip(batch, outs, strict=True):
                results[i] = res
        return results  # type: ignore[return-value]

    def _generate_batch(self, reqs: Sequence[GenRequest], prompts: Sequence[str]) -> list[GenResult]:
        torch = self._torch
        dec = reqs[0].decoding
        enc = self.tokenizer(list(prompts), return_tensors="pt", padding=True, add_special_tokens=False)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        kwargs: dict[str, Any] = {"generation_config": self.generation_config(dec)}
        if self._use_model_defaults_kw:
            kwargs["use_model_defaults"] = False
        if not dec.is_greedy:
            seeds = [effective_seed(r, p) for r, p in zip(reqs, prompts, strict=True)]
            proc = make_gumbel_processor(
                seeds,  # type: ignore[arg-type]
                dec.temperature,
                top_p=dec.top_p,
                top_k=dec.top_k,
                device=self.device,
            )
            kwargs["logits_processor"] = self._transformers.LogitsProcessorList([proc])
        # Defend against anything that mutated the model's generation_config after construction.
        self.model.generation_config = self._neutral_config
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0 / len(reqs)
        prompt_len = input_ids.shape[1]
        new_tokens = out[:, prompt_len:].tolist()
        n_prompt = attention_mask.sum(dim=1).tolist()
        stop = set(self.stop_ids)
        results = []
        for row, ids in enumerate(new_tokens):
            cut = next((j for j, t in enumerate(ids) if t in stop), None)
            if cut is None:
                completion = ids
                n_completion = len(ids)
                finish = "length" if len(ids) >= dec.max_new_tokens else "stop"
            else:
                completion = ids[:cut]
                n_completion = cut + 1  # the EOS token was generated too
                finish = "stop"
            results.append(
                GenResult(
                    text=self.tokenizer.decode(completion, skip_special_tokens=True),
                    finish_reason=finish,
                    n_prompt_tokens=int(n_prompt[row]),
                    n_completion_tokens=int(n_completion),
                    latency_ms=latency_ms,
                )
            )
        return results

    def close(self) -> None:
        self.model = None
        if self.device_type == "cuda":
            self._torch.cuda.empty_cache()
