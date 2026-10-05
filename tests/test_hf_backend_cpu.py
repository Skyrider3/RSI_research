"""Real HF transformers backend on CPU (needs torch + transformers; skipped otherwise).

Fast tests use a ~5 MB random-weight Qwen2.5-architecture model whose tokenizer is the real Qwen2.5
tokenizer (chat template, <|im_end|>) and whose generation_config.json ships Qwen's sampling defaults
(do_sample=True, T=0.7, top_k=20, top_p=0.8, repetition_penalty=1.05) - exactly the hazard the backend must
neutralise. The model is downloaded from the Hub once; without network these tests are skipped.

The slow test loads Qwen2.5-0.5B-Instruct (~1 GB) and is deselected in CI (``-m "not slow"``).
"""

from __future__ import annotations

import copy
import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from driftlab.backends.base import GenRequest  # noqa: E402
from driftlab.backends.gumbel import make_gumbel_processor  # noqa: E402
from driftlab.backends.hf import HFBackend  # noqa: E402
from driftlab.config import REPO_ROOT, HFSection  # noqa: E402
from driftlab.environments import Decoding  # noqa: E402

# The task originally named "hf-internal-testing/tiny-random-Qwen2ForCausalLM", which does not exist on the
# Hub; this TRL test model is the closest equivalent (Qwen2ForCausalLM, Qwen2.5 tokenizer), pinned by commit.
TINY_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
TINY_REV = "ce8d0bf270b28c8fab026ced69fe8aa14b0f0eda"
MINIMAL_CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
SYSTEM = "You are a helpful assistant. Put the final answer in \\boxed{}."
QUESTIONS = [
    "What is 2 + 2?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    "Hi",
]
N_NEW = 24
GREEDY = Decoding(id="greedy", temperature=0.0, max_new_tokens=N_NEW)
T02 = Decoding(id="t02", temperature=0.2, max_new_tokens=N_NEW)


@pytest.fixture(scope="module")
def tiny() -> HFBackend:
    """Loaded through the real loading path; asks for flash_attention_2 to exercise the attn fallback."""
    try:
        backend = HFBackend(
            TINY_ID,
            TINY_REV,
            HFSection(batch_size=8, device="cpu", attn_impl="flash_attention_2"),
            dtype="auto",
        )
    except OSError as e:  # no network / Hub unavailable
        pytest.skip(f"cannot download {TINY_ID}@{TINY_REV}: {e}")
    if not getattr(backend.tokenizer, "chat_template", None):
        backend.tokenizer.chat_template = MINIMAL_CHAT_TEMPLATE
    return backend


def _reqs(dec: Decoding, seeds: list[int | None] | None = None) -> list[GenRequest]:
    seeds = seeds or [None] * len(QUESTIONS)
    return [GenRequest(SYSTEM, q, dec, seed=s) for q, s in zip(QUESTIONS, seeds, strict=True)]


def _argmax_decode(backend: HFBackend, system: str, user: str, max_new: int) -> list[int]:
    """Reference greedy decoding written by hand: argmax of the raw logits, KV cache, stop on EOS."""
    ids = backend.tokenizer(backend.render(system, user), return_tensors="pt", add_special_tokens=False)
    ids = ids["input_ids"].to(backend.device)
    stop = set(backend.stop_ids)
    out: list[int] = []
    with torch.inference_mode():
        res = backend.model(input_ids=ids, use_cache=True)
        for _ in range(max_new):
            nxt = int(res.logits[0, -1].float().argmax())
            out.append(nxt)
            if nxt in stop:
                break
            res = backend.model(
                input_ids=torch.tensor([[nxt]], device=backend.device),
                past_key_values=res.past_key_values,
                use_cache=True,
            )
    return out


# --------------------------------------------------------------------------- engine info / loading


def test_engine_info_cpu(tiny):
    info = tiny.engine_info()
    assert info["kind"] == "hf"
    assert info["dtype"] == "float32"  # auto on CPU
    assert info["device"] == "cpu" and info["gpu_class"] == "cpu" and info["capability"] is None
    assert info["attn"] in ("sdpa", "eager")  # flash_attention_2 unavailable -> fallback
    assert info["batch_size"] == 8
    assert info["stop_token_ids"] == [151645, 151643]  # <|im_end|>, <|endoftext|>
    assert info["transformers"] == transformers.__version__ and info["torch"] == torch.__version__
    json.dumps(info)
    assert tiny.tokenizer.padding_side == "left"
    assert tiny.tokenizer.pad_token is not None
    assert next(tiny.model.parameters()).dtype == torch.float32


def test_missing_pad_token_defaults_to_endoftext(tiny):
    tok = copy.deepcopy(tiny.tokenizer)
    tok.pad_token = None
    b = HFBackend(TINY_ID, TINY_REV, HFSection(device="cpu"), model=tiny.model, tokenizer=tok)
    assert tok.pad_token == "<|endoftext|>" and b.pad_token_id == 151643


def test_render_uses_chat_template(tiny):
    r = tiny.render("SYS", "USER")
    assert r.startswith("<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nUSER<|im_end|>\n")
    assert r.endswith("<|im_start|>assistant\n")
    assert tiny.render("SYS", "USER") is r  # cached


# --------------------------------------------------------------------------- (a) greedy repeatability


def test_greedy_generate_twice_identical(tiny):
    r1 = tiny.generate(_reqs(GREEDY))
    r2 = tiny.generate(_reqs(GREEDY))
    assert [r.text for r in r1] == [r.text for r in r2]
    assert [r.n_completion_tokens for r in r1] == [r.n_completion_tokens for r in r2]
    for r in r1:
        assert r.finish_reason in ("stop", "length")
        assert 0 < r.n_completion_tokens <= N_NEW
        assert r.n_prompt_tokens > 0 and r.latency_ms >= 0
        if r.finish_reason == "length":
            assert r.n_completion_tokens == N_NEW


def test_greedy_independent_of_request_order(tiny):
    reqs = _reqs(GREEDY)
    fwd = tiny.generate(reqs)
    rev = tiny.generate(reqs[::-1])[::-1]
    assert [r.text for r in fwd] == [r.text for r in rev]


# --------------------------------------------------------------------------- (b), (c) per-row seeded sampling


def test_sampling_exact_per_request_seeding_independent_of_batch(tiny):
    a, b, c = _reqs(T02, seeds=[11, 22, 33])
    assert len(tiny.plan_batches([a, b, c])) == 1  # a, b, c really share one padded batch
    abc = tiny.generate([a, b, c])
    ca = tiny.generate([c, a])
    assert abc[0].text == ca[1].text  # row a unchanged
    assert abc[2].text == ca[0].text  # row c unchanged
    assert abc[0].n_completion_tokens == ca[1].n_completion_tokens
    # also independent of the batch size: one request per batch, same model and tokenizer
    single = HFBackend(
        TINY_ID, TINY_REV, HFSection(batch_size=1, device="cpu"), model=tiny.model, tokenizer=tiny.tokenizer
    )
    alone = single.generate([a, b, c])
    assert [r.text for r in alone] == [r.text for r in abc]


def test_sampling_seed_controls_sample(tiny):
    q = QUESTIONS[0]
    reqs = [GenRequest(SYSTEM, q, T02, seed=s) for s in (1, 2, 1)]  # same prompt, seeds 1, 2, 1 in one batch
    res = tiny.generate(reqs)
    assert res[0].text == res[2].text  # same seed -> same sample, even within one batch
    assert res[0].text != res[1].text  # different seed -> different sample
    again = tiny.generate([reqs[1]])
    assert again[0].text == res[1].text  # repeatable across calls
    texts = {r.text for r in tiny.generate([GenRequest(SYSTEM, q, T02, seed=s) for s in range(100, 106)])}
    assert len(texts) == 6


def test_sampling_differs_from_greedy(tiny):
    g = tiny.generate(_reqs(GREEDY))
    s = tiny.generate(_reqs(T02, seeds=[5, 6, 7]))
    assert [r.text for r in g] != [r.text for r in s]


# --------------------------------------------------------------------------- (d) model generation defaults ignored


def test_model_generation_config_sampling_defaults_are_ignored(tiny):
    # the checkpoint really ships Qwen-style sampling defaults ...
    assert tiny.model_generation_defaults.get("do_sample") is True
    assert tiny.model_generation_defaults.get("temperature") == 0.7
    # ... and even if someone re-installs aggressive sampling defaults after construction:
    bad = transformers.GenerationConfig(
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.3,
        eos_token_id=tiny.stop_ids,
        pad_token_id=tiny.pad_token_id,
    )
    tiny.model.generation_config = bad
    ids = tiny.tokenizer(tiny.render(SYSTEM, QUESTIONS[1]), return_tensors="pt", add_special_tokens=False)
    torch.manual_seed(0)
    with torch.inference_mode():
        naive = tiny.model.generate(**ids, max_new_tokens=N_NEW)  # what a naive call would do: sampling
    naive_new = naive[0, ids["input_ids"].shape[1] :].tolist()

    for q in QUESTIONS:
        (res,) = tiny.generate([GenRequest(SYSTEM, q, GREEDY)])
        ref = _argmax_decode(tiny, SYSTEM, q, N_NEW)
        stop = set(tiny.stop_ids)
        ref_text_ids = [t for t in ref if t not in stop]
        assert res.text == tiny.tokenizer.decode(ref_text_ids, skip_special_tokens=True)
        assert res.n_completion_tokens == len(ref)
        assert res.finish_reason == ("stop" if ref[-1] in stop else "length")
    # the hazard is real: honouring the model's defaults would not have been greedy
    assert naive_new != _argmax_decode(tiny, SYSTEM, QUESTIONS[1], N_NEW)
    # the backend restored its neutral config
    assert tiny.model.generation_config.do_sample is False


def test_generation_config_is_fully_explicit(tiny):
    gc = tiny.generation_config(Decoding(id="greedy", temperature=0.0, max_new_tokens=640))
    assert gc.do_sample is False and gc.num_beams == 1
    assert gc.temperature is None and gc.top_p is None and gc.top_k is None
    assert gc.repetition_penalty == 1.0 and gc.max_new_tokens == 640
    assert list(gc.eos_token_id) == [151645, 151643]
    assert gc.pad_token_id == tiny.pad_token_id


# --------------------------------------------------------------------------- finish reasons / token counts


def test_finish_reason_and_token_counts_from_tensors(tiny, monkeypatch):
    eos, endoftext = tiny.stop_ids
    pad = tiny.pad_token_id
    word = tiny.tokenizer.convert_tokens_to_ids("Hello")

    def fake_generate(input_ids, attention_mask, **kwargs):
        assert kwargs["generation_config"].max_new_tokens == 4
        n = input_ids.shape[0]
        rows = [[word, eos, pad, pad], [word, word, word, word], [word, word, endoftext, pad]][:n]
        return torch.cat([input_ids, torch.tensor(rows, device=input_ids.device)], dim=1)

    monkeypatch.setattr(tiny.model, "generate", fake_generate)
    dec = Decoding(id="greedy", temperature=0.0, max_new_tokens=4)
    reqs = [GenRequest(SYSTEM, "same length a", dec), GenRequest(SYSTEM, "same length b", dec)]
    reqs.append(GenRequest(SYSTEM, "same length c", dec))
    res = tiny.generate(reqs)
    assert [r.finish_reason for r in res] == ["stop", "length", "stop"]
    assert [r.n_completion_tokens for r in res] == [2, 4, 3]
    assert res[0].text == "Hello"
    assert res[1].text == "HelloHelloHelloHello"


# --------------------------------------------------------------------------- Gumbel processor distribution


def _gumbel_freqs(logits: list[float], temperature: float, n_rows: int, n_steps: int, **kw) -> torch.Tensor:
    proc = make_gumbel_processor(list(range(1000, 1000 + n_rows)), temperature, device="cpu", **kw)
    scores = torch.tensor([logits] * n_rows, dtype=torch.float32)
    counts = torch.zeros(len(logits))
    for _ in range(n_steps):
        picks = proc(None, scores).argmax(dim=-1)
        counts += torch.bincount(picks, minlength=len(logits)).float()
    assert proc.steps == n_steps
    return counts / counts.sum()


def test_gumbel_processor_is_exact_categorical_sampling():
    logits = [2.0, 1.0, 0.5, 0.0, -1.0]
    freqs = _gumbel_freqs(logits, 0.5, n_rows=2000, n_steps=15)  # 30k samples
    expected = torch.softmax(torch.tensor(logits) / 0.5, dim=-1)
    assert torch.allclose(freqs, expected, atol=0.012)


def test_gumbel_processor_top_k_and_top_p():
    logits = [2.0, 1.0, 0.5, 0.0, -1.0]
    p = torch.softmax(torch.tensor(logits), dim=-1)
    fk = _gumbel_freqs(logits, 1.0, n_rows=2000, n_steps=10, top_k=2)
    assert fk[2:].sum() == 0
    assert torch.allclose(fk[:2], p[:2] / p[:2].sum(), atol=0.015)
    # top_p=0.8 keeps the 3 most likely tokens (mass 0.563 + 0.207 + 0.126 >= 0.8)
    fp = _gumbel_freqs(logits, 1.0, n_rows=2000, n_steps=10, top_p=0.8)
    assert fp[3:].sum() == 0
    assert torch.allclose(fp[:3], p[:3] / p[:3].sum(), atol=0.015)


def test_gumbel_processor_rows_are_independent_streams():
    scores = torch.zeros(3, 50)
    p1 = make_gumbel_processor([7, 8, 9], 0.2)
    p2 = make_gumbel_processor([9, 7], 0.2)
    out1 = [p1(None, scores) for _ in range(3)]
    out2 = [p2(None, scores[:2]) for _ in range(3)]
    for a, b in zip(out1, out2, strict=True):
        assert torch.equal(a[0], b[1]) and torch.equal(a[2], b[0])
    with pytest.raises(ValueError, match="rows"):
        p1(None, torch.zeros(2, 50))
    with pytest.raises(ValueError, match="temperature"):
        make_gumbel_processor([1], 0.0)


# --------------------------------------------------------------------------- slow: real Qwen2.5-0.5B on CPU


@pytest.mark.slow
def test_qwen25_05b_cpu_greedy_smoke():
    """Qwen2.5-0.5B-Instruct @ pinned revision, 2 GSM8K test questions, 48 new tokens (violates the
    640-token protocol on purpose), greedy twice -> identical; 48 tokens truncate at least one answer."""
    from driftlab.backends import make_backend
    from driftlab.config import load_config
    from driftlab.prompting import load_prompt_file

    cfg = load_config(
        REPO_ROOT / "configs" / "smoke_hf_cpu.yaml",
        overrides=[
            "model.id=Qwen/Qwen2.5-0.5B-Instruct",
            "model.revision='7ae557604adf67be50417f59c2c2f167def9a775'",
            "model.max_new_tokens=48",
            "model.dtype=float32",
            "backend.kind=hf",
            "backend.hf.device=cpu",
            "backend.hf.batch_size=2",
        ],
    )
    backend = make_backend(cfg)
    assert isinstance(backend, HFBackend)
    lines = (REPO_ROOT / "data" / "gsm8k" / "test_first200.jsonl").read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line)["question"] for line in lines[:2]]
    system = load_prompt_file(cfg.trajectory.initial_prompt_file)
    dec = cfg.decoding("greedy")
    assert dec.max_new_tokens == 48
    reqs = [GenRequest(system, q, dec) for q in questions]
    r1 = backend.generate(reqs)
    r2 = backend.generate(reqs)
    assert [r.text for r in r1] == [r.text for r in r2]
    assert any(r.finish_reason == "length" for r in r1)
    assert all(r.n_completion_tokens <= 48 for r in r1)
    assert all(r.text.strip() for r in r1)
    info = backend.engine_info()
    assert info["dtype"] == "float32" and info["device"] == "cpu"
