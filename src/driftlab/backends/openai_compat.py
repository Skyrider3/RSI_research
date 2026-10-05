"""OpenAI-compatible HTTP backend (``vllm serve``, the dashboard's live rerun, other servers).

Two rendering modes:

* ``chat_template`` (default when transformers is importable): the prompt is rendered locally with the
  model tokenizer's chat template (tokenizer only, no torch) and sent to ``POST {base_url}/completions``,
  so the server applies no template of its own and ``rendered_sha`` hashes exactly the text the model sees.
* ``messages_json`` (transformers unavailable): requests go to ``POST {base_url}/chat/completions`` and
  ``render()`` returns the canonical JSON of the messages, so ``rendered_sha`` then hashes the *messages*,
  not the template output (the server's template is outside our control and must be held fixed).

``base_url`` includes the API prefix, e.g. ``http://localhost:8000/v1``. Every decoding parameter is sent
explicitly (temperature, top_p, max_tokens, n=1, seed for sampling only) plus the vLLM extensions
``top_k`` (-1 = disabled) / ``min_p`` (0 = disabled) / ``repetition_penalty`` / ``stop_token_ids`` and, in
``chat_template`` mode, ``add_special_tokens: false`` (the rendered prompt already is the exact model input;
vLLM's ``/completions`` would otherwise prepend a BOS for tokenizers that add one, unlike the HF backend).
Disable them with ``vllm_extras=False``; ``extra_body`` is merged last. vLLM's server fills every *omitted*
sampling field (including ``min_p``) from the checkpoint's generation_config.json, so all of them are sent.

Server flags the client cannot enforce (launch vLLM accordingly; they are not visible in ``engine_info``):
``vllm serve ... --generation-config vllm --no-enable-prefix-caching``. vLLM V1 enables prefix caching by
default, which would let a physical rerun reuse the KV cache of the stored generation and bias measured
rerun drift towards zero (the offline vLLM backend disables it for the same reason).

Requests run concurrently (``asyncio.Semaphore(concurrency)``) with deterministic exponential backoff on
429 / 5xx / transport errors (``Retry-After`` honoured). Exhausted retries and non-retryable HTTP errors
yield ``GenResult(finish_reason="error")`` (never cached by the engine). ``generate`` is synchronous and
also works when an event loop is already running (Streamlit / Jupyter): it then runs the async client in a
fresh thread.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections.abc import Callable, Coroutine, Mapping, Sequence
from typing import Any, Literal, TypeVar

import httpx

from driftlab.backends.base import Backend, GenRequest, GenResult
from driftlab.backends.hf import RenderCache, effective_seed, render_chat, stop_token_ids
from driftlab.config import OpenAICompatSection

RenderMode = Literal["auto", "chat_template", "messages_json"]
RETRY_STATUS: frozenset[int] = frozenset({408, 409, 425, 429})
VLLM_TOP_K_DISABLED = -1
VLLM_MIN_P_DISABLED = 0.0
# Bump whenever the request body semantics change (it is in engine_info, hence in every gen_key).
PAYLOAD_VERSION = "v2"  # v2: explicit min_p; add_special_tokens=false for chat_template prompts
_T = TypeVar("_T")


def messages_json(system: str, user: str) -> str:
    """Canonical JSON of the chat messages (the ``render`` output in ``messages_json`` mode)."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def run_sync(factory: Callable[[], Coroutine[Any, Any, _T]]) -> _T:
    """Run ``factory()`` to completion from sync code, even inside a running event loop (fresh thread)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = asyncio.run(factory())
        except BaseException as e:  # re-raised in the caller's thread
            box["error"] = e

    t = threading.Thread(target=target, name="driftlab-openai-compat", daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def _transformers_available() -> bool:
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _map_finish(reason: Any) -> str | None:
    if reason in (None, "stop", "eos", "eos_token", "stop_sequence"):
        return "stop"
    if reason == "length":
        return "length"
    return None


class OpenAICompatBackend(Backend):
    """Client for an OpenAI-compatible completions server.

    Args:
        model_id, model_revision: the model the server is serving (tokenizer loaded at this revision).
        settings: ``cfg.backend.openai_compat`` (base_url, api_key_env, concurrency, timeout_s,
            served_model_name).
        render_mode: ``auto`` (chat_template if transformers is importable, else messages_json),
            ``chat_template`` or ``messages_json``.
        tokenizer: optional pre-loaded tokenizer (implies ``chat_template``).
        vllm_extras: send ``top_k`` / ``repetition_penalty`` / ``stop_token_ids`` (vLLM extensions).
        extra_body: extra JSON fields merged into every payload last (override anything).
        max_retries, backoff_s, max_backoff_s: retry policy (delay = min(max, backoff * 2**attempt)).
        transport: optional ``httpx.AsyncBaseTransport`` (tests use ``httpx.MockTransport``).
        api_key: overrides the key read from the ``api_key_env`` environment variable.
    """

    kind = "openai_compat"

    def __init__(
        self,
        model_id: str,
        model_revision: str,
        settings: OpenAICompatSection | None = None,
        *,
        render_mode: RenderMode = "auto",
        tokenizer: Any = None,
        vllm_extras: bool = True,
        extra_body: Mapping[str, Any] | None = None,
        max_retries: int = 5,
        backoff_s: float = 1.0,
        max_backoff_s: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(model_id, model_revision)
        self.settings = settings or OpenAICompatSection()
        if not self.settings.base_url:
            raise ValueError("backend.openai_compat.base_url is required (e.g. http://localhost:8000/v1)")
        self.base_url = self.settings.base_url.rstrip("/")
        self.served_model = self.settings.served_model_name or model_id
        self.concurrency = max(1, int(self.settings.concurrency))
        self.timeout_s = float(self.settings.timeout_s)
        self.vllm_extras = bool(vllm_extras)
        self.extra_body = dict(extra_body or {})
        self.max_retries = max(0, int(max_retries))
        self.backoff_s = float(backoff_s)
        self.max_backoff_s = float(max_backoff_s)
        self._transport = transport
        key = api_key if api_key is not None else os.environ.get(self.settings.api_key_env or "", "")
        self._api_key = key or None

        if tokenizer is not None and render_mode == "auto":
            render_mode = "chat_template"
        if render_mode == "auto":
            render_mode = "chat_template" if _transformers_available() else "messages_json"
        if render_mode not in ("chat_template", "messages_json"):
            raise ValueError(f"unknown render_mode {render_mode!r}")
        self.render_mode: str = render_mode
        self.tokenizer = None
        self.stop_ids: list[int] = []
        if render_mode == "chat_template":
            if tokenizer is None:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision)
            self.tokenizer = tokenizer
            self.stop_ids = stop_token_ids(tokenizer)
        self.endpoint = "completions" if render_mode == "chat_template" else "chat/completions"
        self.url = f"{self.base_url}/{self.endpoint}"
        self._renders = RenderCache()

    # ------------------------------------------------------------------ protocol
    def engine_info(self) -> dict:
        return {
            "kind": "openai_compat",
            # base_url is deliberately NOT fingerprinted (tunnel URLs change between Colab sessions);
            # the user-declared server_tag identifies the server-side engine instead.
            "server_tag": getattr(self.settings, "server_tag", ""),
            "served_model": self.served_model,
            "render": self.render_mode,
            "endpoint": self.endpoint,
            "payload": PAYLOAD_VERSION,
            "vllm_extras": self.vllm_extras,
            "extra_body": json.dumps(self.extra_body, sort_keys=True, default=str),
        }

    def render(self, system: str, user: str) -> str:
        key = (system, user)
        out = self._renders.get(key)
        if out is None:
            if self.render_mode == "chat_template":
                out = render_chat(self.tokenizer, system, user)
            else:
                out = messages_json(system, user)
            self._renders.put(key, out)
        return out

    def payload(self, req: GenRequest) -> dict[str, Any]:
        """JSON body for one request (exposed for tests and debugging)."""
        d = req.decoding
        rendered = self.render(req.system, req.user)
        body: dict[str, Any] = {"model": self.served_model}
        if self.render_mode == "chat_template":
            body["prompt"] = rendered
        else:
            body["messages"] = [
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.user},
            ]
        body.update(
            {
                "temperature": float(d.temperature),
                "top_p": float(d.top_p),
                "max_tokens": int(d.max_new_tokens),
                "n": 1,
            }
        )
        seed = effective_seed(req, rendered)
        if seed is not None:
            body["seed"] = seed
        if self.vllm_extras:
            body["top_k"] = int(d.top_k) if d.top_k > 0 else VLLM_TOP_K_DISABLED
            body["min_p"] = VLLM_MIN_P_DISABLED  # omitted -> server falls back to generation_config.json
            body["repetition_penalty"] = float(d.repetition_penalty)
            if self.stop_ids:
                body["stop_token_ids"] = list(self.stop_ids)
            if self.render_mode == "chat_template":
                body["add_special_tokens"] = False  # the rendered template is the exact model input
        body.update(self.extra_body)
        return body

    def generate(self, reqs: Sequence[GenRequest]) -> list[GenResult]:
        if not reqs:
            return []
        payloads = [self.payload(r) for r in reqs]
        return run_sync(lambda: self._generate_all(payloads))

    # ------------------------------------------------------------------ HTTP
    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _delay(self, attempt: int, response: httpx.Response | None) -> float:
        if response is not None:
            ra = response.headers.get("retry-after")
            if ra is not None:
                try:
                    return min(self.max_backoff_s, max(0.0, float(ra)))
                except ValueError:
                    pass
        return min(self.max_backoff_s, self.backoff_s * (2**attempt))

    async def _generate_all(self, payloads: Sequence[dict[str, Any]]) -> list[GenResult]:
        sem = asyncio.Semaphore(self.concurrency)
        async with httpx.AsyncClient(
            timeout=self.timeout_s, transport=self._transport, headers=self._headers()
        ) as client:
            return list(await asyncio.gather(*(self._one(client, sem, p) for p in payloads)))

    async def _one(
        self, client: httpx.AsyncClient, sem: asyncio.Semaphore, payload: dict[str, Any]
    ) -> GenResult:
        last_error = ""
        for attempt in range(self.max_retries + 1):
            response: httpx.Response | None = None
            t0 = time.perf_counter()
            try:
                async with sem:
                    response = await client.post(self.url, json=payload)
            except httpx.TransportError as e:  # connect errors, timeouts, protocol errors
                last_error = f"{type(e).__name__}: {e}"
            else:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                code = response.status_code
                if code == 200:
                    return self._parse(response, latency_ms, payload.get("max_tokens"))
                last_error = f"HTTP {code}: {response.text[:500]}"
                if code < 500 and code not in RETRY_STATUS:
                    return GenResult(text=f"{self.url} -> {last_error}", finish_reason="error")
            if attempt < self.max_retries:
                await asyncio.sleep(self._delay(attempt, response))
        return GenResult(
            text=f"{self.url} failed after {self.max_retries + 1} attempt(s): {last_error}",
            finish_reason="error",
        )

    def _parse(self, response: httpx.Response, latency_ms: float, max_tokens: Any = None) -> GenResult:
        try:
            data = response.json()
            choice = data["choices"][0]
            if self.render_mode == "chat_template":
                text = choice.get("text") or ""
            else:
                text = (choice.get("message") or {}).get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError) as e:
            return GenResult(text=f"malformed response ({e}): {response.text[:300]}", finish_reason="error")
        finish = _map_finish(choice.get("finish_reason"))
        if finish is None:
            return GenResult(
                text=f"unexpected finish_reason {choice.get('finish_reason')!r}: {text[:200]}",
                finish_reason="error",
            )
        usage = data.get("usage") or {}
        n_completion = int(usage.get("completion_tokens") or 0)
        if choice.get("finish_reason") is None and max_tokens and n_completion >= int(max_tokens):
            finish = "length"  # a null finish_reason must not hide a truncation (truncation is reported)
        return GenResult(
            text=text,
            finish_reason=finish,
            n_prompt_tokens=int(usage.get("prompt_tokens") or 0),
            n_completion_tokens=n_completion,
            latency_ms=latency_ms,
        )
