"""Backend factory, vLLM and OpenAI-compatible backends with fakes; torch-free helpers.

Everything here must pass WITHOUT torch / transformers / vllm installed: vLLM is a fake module injected
into ``sys.modules`` and HTTP goes through ``httpx.MockTransport``.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
import textwrap
import types
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import driftlab.backends as backends
from driftlab.backends import gumbel, openai_compat, vllm_backend
from driftlab.backends.base import GenRequest
from driftlab.backends.hf import (
    LENGTH_BUCKET,
    RenderCache,
    batch_plan,
    effective_seed,
    normalize_dtype_name,
    resolve_dtype_name,
    stop_token_ids,
    transformers_dtype_kwarg,
)
from driftlab.backends.openai_compat import OpenAICompatBackend, messages_json
from driftlab.config import ExperimentConfig, OpenAICompatSection
from driftlab.environments import Decoding

GREEDY = Decoding(id="greedy", temperature=0.0)
T02 = Decoding(id="t02", temperature=0.2)
SYSTEM = "Solve the problem and put the final answer in \\boxed{}."


def _cfg(kind: str, **backend: Any) -> ExperimentConfig:
    return ExperimentConfig.model_validate({"backend": {"kind": kind, **backend}})


# --------------------------------------------------------------------------- fakes


class FakeTokenizer:
    """Minimal HF-tokenizer stand-in: chat template + special-token lookup."""

    eos_token_id = 2
    unk_token_id = None
    _special = {"<|im_end|>": 2, "<|endoftext|>": 3}

    def __init__(self) -> None:
        self.template_calls = 0

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self._special.get(token)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        assert tokenize is False and add_generation_prompt is True
        self.template_calls += 1
        body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return body + "<|im_start|>assistant\n"


def make_fake_vllm(
    *, top_k_default: int = -1, accept_generation_config: bool = True, version: str = "0.6.3.fake"
) -> tuple[types.ModuleType, dict[str, list]]:
    """A fake ``vllm`` module recording LLM(...) kwargs, SamplingParams(...) and generate() calls."""
    record: dict[str, list] = {"llm_kwargs": [], "params": [], "generate": []}
    mod = types.ModuleType("vllm")
    mod.__version__ = version

    class SamplingParams:
        def __init__(
            self,
            n: int = 1,
            temperature: float = 1.0,
            top_p: float = 1.0,
            top_k: int = top_k_default,
            repetition_penalty: float = 1.0,
            max_tokens: int | None = 16,
            seed: int | None = None,
            stop_token_ids: list[int] | None = None,
        ) -> None:
            if top_k_default == -1 and top_k == 0:
                raise ValueError("top_k must be -1 (disable), or at least 1, got 0.")
            if top_k < -1:
                raise ValueError(f"bad top_k {top_k}")
            self.n = n
            self.temperature = temperature
            self.top_p = top_p
            self.top_k = top_k
            self.repetition_penalty = repetition_penalty
            self.max_tokens = max_tokens
            self.seed = seed
            self.stop_token_ids = stop_token_ids

    class LLM:
        def __init__(self, **kwargs: Any) -> None:
            if not accept_generation_config and "generation_config" in kwargs:
                raise TypeError(
                    "EngineArgs.__init__() got an unexpected keyword argument 'generation_config'"
                )
            record["llm_kwargs"].append(dict(kwargs))
            self.tokenizer = FakeTokenizer()
            self.llm_engine = SimpleNamespace(model_config=SimpleNamespace(dtype=f"torch.{kwargs['dtype']}"))

        def get_tokenizer(self) -> FakeTokenizer:
            return self.tokenizer

        def generate(self, prompts, sampling_params=None, use_tqdm=True):
            record["generate"].append({"prompts": list(prompts), "use_tqdm": use_tqdm})
            outs = []
            for p, sp in zip(prompts, sampling_params, strict=True):
                record["params"].append(sp)
                long = "LONG" in p
                n_tok = sp.max_tokens if long else 3
                outs.append(
                    SimpleNamespace(
                        prompt_token_ids=list(range(len(p.split()))),
                        outputs=[
                            SimpleNamespace(
                                text=f"T={sp.temperature} seed={sp.seed}",
                                finish_reason="length" if long else "stop",
                                token_ids=list(range(n_tok)),
                            )
                        ],
                    )
                )
            return outs

    mod.SamplingParams = SamplingParams
    mod.LLM = LLM
    return mod, record


@pytest.fixture
def t4(monkeypatch):
    monkeypatch.setattr(vllm_backend, "_device_info", lambda: {"name": "Tesla T4", "capability": (7, 5)})


@pytest.fixture
def fake_vllm(monkeypatch, t4):
    mod, record = make_fake_vllm()
    monkeypatch.setitem(sys.modules, "vllm", mod)
    return record


# --------------------------------------------------------------------------- import hygiene


def test_backend_modules_import_without_heavy_deps():
    code = textwrap.dedent(
        """
        import sys
        import driftlab.backends, driftlab.backends.gumbel, driftlab.backends.hf
        import driftlab.backends.vllm_backend, driftlab.backends.openai_compat
        heavy = sorted(m for m in ("torch", "transformers", "vllm") if m in sys.modules)
        print(",".join(heavy))
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


def test_gumbel_importable_without_torch_and_fails_clearly_when_used(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setattr(gumbel, "_CLASS", None)
    assert gumbel.SAMPLER_VERSION.startswith("gumbel")
    with pytest.raises(ImportError, match="torch and transformers"):
        gumbel.make_gumbel_processor([1, 2], 0.2)
    with pytest.raises(ImportError, match="torch and transformers"):
        from driftlab.backends.gumbel import PerRowGumbelProcessor  # noqa: F401
    with pytest.raises(AttributeError):
        _ = gumbel.no_such_name


def test_hf_backend_without_torch_raises_clear_import_error(monkeypatch):
    from driftlab.backends.hf import HFBackend

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match=r"driftlab\[hf\]"):
        HFBackend("m", "rev")


def test_vllm_backend_without_vllm_raises_clear_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", None)
    with pytest.raises(ImportError, match="pip install vllm"):
        vllm_backend.VLLMBackend("m", "rev")


# --------------------------------------------------------------------------- factory


def test_make_backend_mock_is_lazy_and_passes_arguments(monkeypatch):
    calls = []

    class FakeMock:
        def __init__(self, model_id, model_revision, settings, answer_key=None):
            calls.append((model_id, model_revision, settings, answer_key))

    fake = types.ModuleType("driftlab.backends.mock")
    fake.MockBackend = FakeMock
    monkeypatch.setitem(sys.modules, "driftlab.backends.mock", fake)
    cfg = _cfg("mock")
    b = backends.make_backend(cfg, answer_key={"test:0": "18"})
    assert isinstance(b, FakeMock)
    assert calls == [(cfg.model.id, cfg.model.revision, cfg.backend.mock, {"test:0": "18"})]


@pytest.mark.skipif(
    importlib.util.find_spec("driftlab.backends.mock") is None,
    reason="driftlab.backends.mock not written yet",
)
def test_make_backend_mock_real():
    from driftlab.backends.mock import MockBackend

    cfg = _cfg("mock")
    b = backends.make_backend(cfg)
    assert isinstance(b, MockBackend)
    assert b.synthetic is True
    assert (b.model_id, b.model_revision) == (cfg.model.id, cfg.model.revision)


def test_auto_without_cuda_raises_helpful_error(monkeypatch):
    monkeypatch.setattr(backends, "cuda_available", lambda: False)
    with pytest.raises(RuntimeError) as ei:
        backends.make_backend(ExperimentConfig())
    msg = str(ei.value)
    for needle in ("mock", "hf", "cpu", "Colab", "openai_compat"):
        assert needle in msg


def test_cuda_available_false_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert backends.cuda_available() is False
    with pytest.raises(RuntimeError, match="no CUDA GPU"):
        backends.resolve_kind("auto")


def test_auto_prefers_vllm_then_hf_when_cuda(monkeypatch):
    monkeypatch.setattr(backends, "cuda_available", lambda: True)
    mod, _ = make_fake_vllm()
    monkeypatch.setitem(sys.modules, "vllm", mod)
    assert backends.resolve_kind("auto") == "vllm"
    monkeypatch.setitem(sys.modules, "vllm", None)  # vllm not importable
    assert backends.resolve_kind("auto") == "hf"


def test_resolve_kind_explicit_and_invalid():
    for k in backends.KINDS:
        assert backends.resolve_kind(k) == k
    with pytest.raises(ValueError, match="unknown backend kind"):
        backends.resolve_kind("tpu")


def test_module_available():
    assert backends.module_available("json")
    assert not backends.module_available("definitely_not_a_module_xyz")


# --------------------------------------------------------------------------- torch-free helpers


@pytest.mark.parametrize(
    ("name", "device", "cap", "expected"),
    [
        ("auto", "cuda", (7, 5), "float16"),  # T4
        ("auto", "cuda", (8, 0), "bfloat16"),  # A100
        ("auto", "cuda:0", (8, 6), "bfloat16"),
        ("auto", "cpu", None, "float32"),
        ("fp16", "cuda", (8, 0), "float16"),  # explicit dtype honoured
        ("bf16", "cuda", (7, 5), "bfloat16"),
        ("torch.float32", "cuda", (8, 0), "float32"),
    ],
)
def test_resolve_dtype_name(name, device, cap, expected):
    assert resolve_dtype_name(name, device, cap) == expected


def test_resolve_dtype_name_errors():
    with pytest.raises(ValueError, match="unknown dtype"):
        normalize_dtype_name("int8")
    with pytest.raises(ValueError, match="capability"):
        resolve_dtype_name("auto", "cuda", None)


@pytest.mark.parametrize(
    ("version", "kw"),
    [
        ("4.46.3", "torch_dtype"),
        ("4.55.4", "torch_dtype"),
        ("4.56.0", "dtype"),
        ("4.57.0.dev0", "dtype"),
        ("5.0.0rc1", "dtype"),
        ("5.18.0", "dtype"),
    ],
)
def test_transformers_dtype_kwarg(version, kw):
    assert transformers_dtype_kwarg(version) == kw


def test_stop_token_ids_dedup_and_missing():
    assert stop_token_ids(FakeTokenizer()) == [2, 3]

    class Tok:
        eos_token_id = 7
        unk_token_id = 0

        def convert_tokens_to_ids(self, token):
            return {"<|im_end|>": 7}.get(token, 0)  # missing tokens map to unk

    assert stop_token_ids(Tok()) == [7]


def test_effective_seed():
    assert effective_seed(GenRequest("s", "u", GREEDY, seed=5), "r") is None
    assert effective_seed(GenRequest("s", "u", T02, seed=5), "r") == 5
    a = effective_seed(GenRequest("s", "u", T02), "prompt-a")
    assert a == effective_seed(GenRequest("s", "u", T02), "prompt-a")
    assert a != effective_seed(GenRequest("s", "u", T02), "prompt-b")
    assert 0 <= a < 2**31


def test_batch_plan_groups_decodings_and_is_deterministic():
    decs = [GREEDY, T02, GREEDY, T02, GREEDY, GREEDY, GREEDY]
    lengths = [10, 50, 100, 20, 12, 5 * LENGTH_BUCKET, 3]
    plan = batch_plan(decs, lengths, batch_size=2)
    assert plan == batch_plan(decs, lengths, batch_size=2)
    assert sorted(i for b in plan for i in b) == list(range(len(decs)))
    for b in plan:
        assert len(b) <= 2
        assert len({decs[i].temperature for i in b}) == 1
    # greedy first (T=0 < 0.2), longest bucket first, ties broken by original index
    assert plan == [[2, 5], [0, 4], [6], [1, 3]]
    # same decoding parameters under different ids share batches
    same = Decoding(id="other", temperature=0.0)
    assert batch_plan([GREEDY, same], [10, 10], batch_size=4) == [[0, 1]]
    with pytest.raises(ValueError):
        batch_plan(decs, lengths, batch_size=0)


def test_render_cache_lru():
    c = RenderCache(maxsize=2)
    c.put(("a", "1"), "A")
    c.put(("b", "2"), "B")
    assert c.get(("a", "1")) == "A"  # refreshes a
    c.put(("c", "3"), "C")  # evicts b
    assert c.get(("b", "2")) is None
    assert c.get(("a", "1")) == "A" and c.get(("c", "3")) == "C"


# --------------------------------------------------------------------------- vLLM (fake module)


def test_vllm_llm_kwargs_and_sampling_params(fake_vllm):
    cfg = _cfg("vllm")
    b = backends.make_backend(cfg)
    assert isinstance(b, vllm_backend.VLLMBackend)
    (kw,) = fake_vllm["llm_kwargs"]
    assert kw["generation_config"] == "vllm"
    assert kw["enable_prefix_caching"] is False
    assert kw["model"] == cfg.model.id
    assert kw["revision"] == kw["tokenizer_revision"] == cfg.model.revision
    assert kw["dtype"] == "float16"  # T4 (sm75), dtype auto
    assert kw["seed"] == 0
    assert kw["max_model_len"] == cfg.backend.vllm.max_model_len
    assert kw["gpu_memory_utilization"] == cfg.backend.vllm.gpu_memory_utilization
    assert kw["max_num_seqs"] == cfg.backend.vllm.max_num_seqs
    assert kw["enforce_eager"] is False

    reqs = [
        GenRequest(SYSTEM, "q1", cfg.decoding("greedy"), seed=123),  # seed must be dropped for greedy
        GenRequest(SYSTEM, "q2 LONG", cfg.decoding("t02"), seed=4242),
        GenRequest(SYSTEM, "q3", cfg.decoding("greedy")),
    ]
    res = b.generate(reqs)
    (call,) = fake_vllm["generate"]
    assert call["use_tqdm"] is False
    assert call["prompts"] == [b.render(r.system, r.user) for r in reqs]
    assert call["prompts"][0].startswith("<|im_start|>system\n" + SYSTEM)
    assert call["prompts"][0].endswith("<|im_start|>assistant\n")
    params = fake_vllm["params"]
    assert len(params) == 3
    for p, r in zip(params, reqs, strict=True):
        assert p.n == 1
        assert p.temperature == r.decoding.temperature
        assert p.top_p == 1.0
        assert p.top_k == -1  # disabled, old-vLLM spelling
        assert p.repetition_penalty == 1.0
        assert p.max_tokens == 640
        assert p.stop_token_ids == [2, 3]
    assert [p.seed for p in params] == [None, 4242, None]
    assert [r.finish_reason for r in res] == ["stop", "length", "stop"]
    assert res[1].n_completion_tokens == 640 and res[0].n_completion_tokens == 3
    assert res[0].n_prompt_tokens > 0

    info = b.engine_info()
    assert info["kind"] == "vllm" and info["vllm"] == "0.6.3.fake"
    assert info["prefix_caching"] is False
    assert info["gpu_class"] == "Tesla T4" and info["capability"] == "7.5"
    assert info["dtype"] == "float16"
    assert info["generation_config"] == "vllm"
    assert info["top_k_disabled"] == -1
    json.dumps(info)  # fingerprintable


def test_vllm_new_version_uses_zero_for_disabled_top_k_and_bf16_on_a100(monkeypatch):
    mod, record = make_fake_vllm(top_k_default=0, version="0.10.1.fake")
    monkeypatch.setitem(sys.modules, "vllm", mod)
    monkeypatch.setattr(
        vllm_backend, "_device_info", lambda: {"name": "NVIDIA A100-SXM4-40GB", "capability": (8, 0)}
    )
    b = vllm_backend.VLLMBackend("m", "rev")
    assert record["llm_kwargs"][0]["dtype"] == "bfloat16"
    b.generate([GenRequest(SYSTEM, "q", T02, seed=1)])
    assert record["params"][0].top_k == 0
    assert b.engine_info()["top_k_disabled"] == 0


def test_vllm_explicit_top_k_and_repetition_penalty(fake_vllm):
    b = vllm_backend.VLLMBackend("m", "rev")
    dec = Decoding(
        id="custom", temperature=0.7, top_p=0.9, top_k=20, repetition_penalty=1.1, max_new_tokens=64
    )
    b.generate([GenRequest(SYSTEM, "q", dec, seed=9)])
    p = fake_vllm["params"][0]
    assert (p.top_k, p.top_p, p.repetition_penalty, p.max_tokens, p.seed) == (20, 0.9, 1.1, 64, 9)


def test_vllm_unseeded_sampling_gets_deterministic_seed(fake_vllm):
    b = vllm_backend.VLLMBackend("m", "rev")
    req = GenRequest(SYSTEM, "q", T02)
    b.generate([req])
    b.generate([req])
    s1, s2 = (p.seed for p in fake_vllm["params"])
    assert s1 is not None and s1 == s2


def test_vllm_old_version_without_generation_config(monkeypatch, t4):
    mod, record = make_fake_vllm(accept_generation_config=False)
    monkeypatch.setitem(sys.modules, "vllm", mod)
    b = vllm_backend.VLLMBackend("m", "rev")
    assert "generation_config" not in record["llm_kwargs"][0]
    assert b.engine_info()["generation_config"].startswith("unsupported")
    b.generate([GenRequest(SYSTEM, "q", GREEDY)])
    p = record["params"][0]
    assert (p.temperature, p.top_p, p.top_k, p.repetition_penalty, p.max_tokens) == (0.0, 1.0, -1, 1.0, 640)


def test_vllm_other_type_errors_propagate(monkeypatch, t4):
    mod, _ = make_fake_vllm()

    class BrokenLLM:
        def __init__(self, **kwargs):
            raise TypeError("something else entirely")

    mod.LLM = BrokenLLM
    monkeypatch.setitem(sys.modules, "vllm", mod)
    with pytest.raises(TypeError, match="something else"):
        vllm_backend.VLLMBackend("m", "rev")


def test_vllm_injected_llm_is_not_vouched_for(fake_vllm):
    llm = sys.modules["vllm"].LLM(model="m", dtype="float16")
    b = vllm_backend.VLLMBackend("m", "rev", llm=llm)
    assert b.llm is llm and len(fake_vllm["llm_kwargs"]) == 1  # the backend built no second engine
    assert b.engine_info()["generation_config"] == "external-llm"
    b.generate([GenRequest(SYSTEM, "q", GREEDY)])
    assert fake_vllm["params"][0].max_tokens == 640


def test_vllm_engine_env_overrides_change_fingerprint(fake_vllm, monkeypatch):
    from driftlab.keys import engine_fingerprint

    for var in vllm_backend.ENGINE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    b = vllm_backend.VLLMBackend("m", "rev")
    info = b.engine_info()
    assert info["env"] == {"VLLM_USE_V1": None, "VLLM_ATTENTION_BACKEND": None}
    assert "transformers" in info and "tokenizers" in info
    monkeypatch.setenv("VLLM_USE_V1", "0")  # V0 engine forced on a resumed session: a different engine
    assert b.engine_info() == info  # captured when the engine was built
    changed = vllm_backend.VLLMBackend("m", "rev").engine_info()
    assert changed["env"]["VLLM_USE_V1"] == "0"
    assert engine_fingerprint(changed) != engine_fingerprint(info)


def test_vllm_prefix_caching_flag_passthrough(fake_vllm):
    cfg = _cfg("vllm", vllm={"enable_prefix_caching": True})
    b = backends.make_backend(cfg)
    assert fake_vllm["llm_kwargs"][0]["enable_prefix_caching"] is True
    assert b.engine_info()["prefix_caching"] is True


# --------------------------------------------------------------------------- OpenAI-compatible (MockTransport)


def _ok(text: str = "The answer is \\boxed{18}.", finish: str = "stop", chat: bool = False) -> dict:
    choice: dict[str, Any] = {"index": 0, "finish_reason": finish}
    if chat:
        choice["message"] = {"role": "assistant", "content": text}
    else:
        choice["text"] = text
    return {"choices": [choice], "usage": {"prompt_tokens": 11, "completion_tokens": 7}}


def _oa(handler, *, mode: str = "messages_json", **kw: Any) -> OpenAICompatBackend:
    settings = OpenAICompatSection(
        base_url="http://server.test/v1/",
        served_model_name="qwen-served",
        concurrency=kw.pop("concurrency", 4),
    )
    tok = FakeTokenizer() if mode == "chat_template" else None
    return OpenAICompatBackend(
        "Qwen/Qwen2.5-1.5B-Instruct",
        "rev",
        settings,
        render_mode=mode,
        tokenizer=tok,
        transport=httpx.MockTransport(handler),
        backoff_s=0.0,
        **kw,
    )


def test_openai_chat_template_payload(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_ok(finish="length" if b"LONG" in request.content else "stop"))

    b = _oa(handler, mode="chat_template")
    reqs = [
        GenRequest(SYSTEM, "q1", Decoding(id="greedy", temperature=0.0), seed=99),
        GenRequest(SYSTEM, "q2 LONG", Decoding(id="t02", temperature=0.2), seed=4242),
    ]
    res = b.generate(reqs)
    assert [r.finish_reason for r in res] == ["stop", "length"]
    assert res[0].text == "The answer is \\boxed{18}."
    assert (res[0].n_prompt_tokens, res[0].n_completion_tokens) == (11, 7)
    assert len(seen) == 2
    bodies = sorted((json.loads(r.content) for r in seen), key=lambda d: d["temperature"])
    for r in seen:
        assert str(r.url) == "http://server.test/v1/completions"
        assert r.headers["authorization"] == "Bearer sk-test"
    g, s = bodies
    assert g["model"] == s["model"] == "qwen-served"
    assert g["prompt"] == b.render(SYSTEM, "q1")
    assert g["prompt"].endswith("<|im_start|>assistant\n")
    for body in bodies:
        assert body["top_p"] == 1.0 and body["max_tokens"] == 640 and body["n"] == 1
        assert body["top_k"] == -1 and body["repetition_penalty"] == 1.0
        # vLLM fills every omitted sampling field (min_p too) from the checkpoint's generation_config
        assert body["min_p"] == 0.0
        # the rendered template is the exact model input: no extra BOS from /completions
        assert body["add_special_tokens"] is False
        assert body["stop_token_ids"] == [2, 3]
    assert g["temperature"] == 0.0 and "seed" not in g  # greedy: no seed
    assert s["temperature"] == 0.2 and s["seed"] == 4242
    info = b.engine_info()
    assert info["kind"] == "openai_compat" and info["render"] == "chat_template"
    assert "base_url" not in info and info["served_model"] == "qwen-served"
    assert info["server_tag"] == ""
    assert info["payload"] == openai_compat.PAYLOAD_VERSION  # body semantics are fingerprinted


def test_openai_messages_json_mode(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_ok("hi", chat=True))

    b = _oa(handler)
    assert b.render(SYSTEM, "q") == messages_json(SYSTEM, "q")
    assert json.loads(b.render(SYSTEM, "q")) == [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "q"},
    ]
    (res,) = b.generate([GenRequest(SYSTEM, "q", GREEDY)])
    assert res.text == "hi" and res.finish_reason == "stop"
    (r,) = seen
    assert str(r.url) == "http://server.test/v1/chat/completions"
    assert "authorization" not in r.headers
    body = json.loads(r.content)
    assert body["messages"][1] == {"role": "user", "content": "q"}
    assert "prompt" not in body and "seed" not in body and "stop_token_ids" not in body
    assert "add_special_tokens" not in body  # chat endpoint: the server's template decides
    assert body["min_p"] == 0.0 and body["top_k"] == -1 and body["repetition_penalty"] == 1.0
    assert b.engine_info()["render"] == "messages_json"


def test_openai_retries_500_then_succeeds():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(500, text="overloaded")
        return httpx.Response(200, json=_ok("ok", chat=True))

    (res,) = _oa(handler).generate([GenRequest(SYSTEM, "q", T02, seed=3)])
    assert len(calls) == 2
    assert res.finish_reason == "stop" and res.text == "ok"


def test_openai_retries_429_and_transport_errors():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, text="slow down")
        if len(calls) == 2:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, json=_ok("ok", chat=True))

    (res,) = _oa(handler).generate([GenRequest(SYSTEM, "q", GREEDY)])
    assert len(calls) == 3 and res.finish_reason == "stop"


def test_openai_non_retryable_and_exhausted_errors():
    calls = []

    def bad_request(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, text="top_k must be ...")

    (res,) = _oa(bad_request).generate([GenRequest(SYSTEM, "q", GREEDY)])
    assert len(calls) == 1
    assert res.finish_reason == "error" and "HTTP 400" in res.text

    def always_503(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    (res,) = _oa(always_503, max_retries=2).generate([GenRequest(SYSTEM, "q", GREEDY)])
    assert res.finish_reason == "error" and "after 3 attempt(s)" in res.text


def test_openai_malformed_and_unexpected_finish_reason():
    def handler(request: httpx.Request) -> httpx.Response:
        if b"q1" in request.content:
            return httpx.Response(200, text="not json")
        return httpx.Response(200, json=_ok("x", finish="abort", chat=True))

    res = _oa(handler).generate([GenRequest(SYSTEM, "q1", GREEDY), GenRequest(SYSTEM, "q2", GREEDY)])
    assert [r.finish_reason for r in res] == ["error", "error"]


def test_openai_bounded_concurrency_and_order():
    state = {"inflight": 0, "max": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        state["inflight"] += 1
        state["max"] = max(state["max"], state["inflight"])
        await asyncio.sleep(0.01)
        state["inflight"] -= 1
        user = json.loads(request.content)["messages"][1]["content"]
        return httpx.Response(200, json=_ok(f"echo {user}", chat=True))

    b = _oa(handler, concurrency=3)
    reqs = [GenRequest(SYSTEM, f"q{i}", GREEDY) for i in range(12)]
    res = b.generate(reqs)
    assert [r.text for r in res] == [f"echo q{i}" for i in range(12)]
    assert 1 <= state["max"] <= 3


def test_openai_generate_inside_running_event_loop():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok("from thread", chat=True))

    b = _oa(handler)

    async def main() -> list:
        asyncio.get_running_loop()  # a loop is running here (Streamlit / Jupyter situation)
        return b.generate([GenRequest(SYSTEM, "q", GREEDY)])

    (res,) = asyncio.run(main())
    assert res.text == "from thread"


def test_openai_extras_toggle_and_extra_body():
    b = _oa(lambda r: httpx.Response(200, json=_ok(chat=True)), vllm_extras=False, extra_body={"min_p": 0.0})
    body = b.payload(GenRequest(SYSTEM, "q", T02, seed=1))
    assert "top_k" not in body and "repetition_penalty" not in body
    assert body["min_p"] == 0.0 and body["seed"] == 1  # min_p here comes from extra_body only
    plain = _oa(lambda r: httpx.Response(200, json=_ok(chat=True)), vllm_extras=False)
    assert not {"top_k", "min_p", "repetition_penalty", "add_special_tokens"} & set(
        plain.payload(GenRequest(SYSTEM, "q", T02, seed=1))
    )


def test_openai_null_finish_reason_does_not_hide_truncation():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        n = body["max_tokens"] if "LONG" in body["messages"][1]["content"] else 3
        data = _ok("x", finish=None, chat=True)
        data["usage"]["completion_tokens"] = n
        return httpx.Response(200, json=data)

    dec = Decoding(id="greedy", temperature=0.0, max_new_tokens=32)
    res = _oa(handler).generate([GenRequest(SYSTEM, "q LONG", dec), GenRequest(SYSTEM, "q", dec)])
    assert [r.finish_reason for r in res] == ["length", "stop"]
    assert res[0].n_completion_tokens == 32


def test_openai_requires_base_url_and_factory(monkeypatch):
    with pytest.raises(ValueError, match="base_url"):
        OpenAICompatBackend("m", "rev", OpenAICompatSection())
    monkeypatch.setattr(openai_compat, "_transformers_available", lambda: False)
    cfg = _cfg("openai_compat", openai_compat={"base_url": "http://localhost:8000/v1"})
    b = backends.make_backend(cfg)
    assert isinstance(b, OpenAICompatBackend)
    assert b.render_mode == "messages_json" and b.served_model == cfg.model.id
    assert b.url == "http://localhost:8000/v1/chat/completions"
