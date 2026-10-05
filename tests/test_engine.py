"""GenerationEngine: caching, dedupe, physical reruns, ledger accounting, budgets and crash-resume."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from driftlab.backends.base import Backend, GenRequest, GenResult
from driftlab.engine import (
    FAIL_AFTER_ENV,
    BudgetExhausted,
    CellKey,
    CellTask,
    GenerationEngine,
    GenerationError,
    GenRecord,
    InjectedFailure,
)
from driftlab.environments import Decoding
from driftlab.keys import engine_fingerprint, proposer_seed, sample_seed, sha256_text
from driftlab.prompting import PROPOSER_SYSTEM
from driftlab.store import ShardWriter, Store, replay_shards

RUN = "run_eng"
GREEDY = Decoding(id="greedy", temperature=0.0)
T02 = Decoding(id="t02", temperature=0.2)
PROPOSER = Decoding(id="proposer", temperature=0.7)


class FakeBackend(Backend):
    """Deterministic test double: output = f(sha256(rendered, seed if sampling, nonce))."""

    kind = "fake"
    synthetic = True

    def __init__(self, revision: str = "r1", engine_tag: str = "e1", fail_users: set[str] | None = None):
        super().__init__("fake/model", revision)
        self.engine_tag = engine_tag
        self.fail_users = set(fail_users or ())
        self.calls = 0  # generate() invocations
        self.n_generated = 0  # requests served
        self.seen: list[GenRequest] = []

    def engine_info(self) -> dict:
        return {"kind": self.kind, "engine": self.engine_tag}

    def render(self, system: str, user: str) -> str:
        return system + "\n" + user

    def generate(self, reqs: Sequence[GenRequest]) -> list[GenResult]:
        self.calls += 1
        out = []
        for r in reqs:
            self.n_generated += 1
            self.seen.append(r)
            if r.user in self.fail_users:
                out.append(GenResult(text="CUDA error: device-side assert", finish_reason="error"))
                continue
            seed = None if r.decoding.is_greedy else r.seed
            h = sha256_text(f"{self.render(r.system, r.user)}|{seed}|{r.nonce}")
            n = int(h[:8], 16) % 1000
            out.append(
                GenResult(
                    text=f"Work {h[8:14]}. The answer is \\boxed{{{n}}}",
                    finish_reason="length" if n % 7 == 0 else "stop",
                    n_prompt_tokens=len(r.system + r.user) // 4,
                    n_completion_tokens=3 + n % 5,
                    latency_ms=0.25,
                )
            )
        return out


def task(
    item: int,
    *,
    slot: int = 0,
    draw: int = 0,
    seed: int = 0,
    dec: Decoding = GREEDY,
    req_seed: int | None = None,
    nonce: str | None = None,
    system: str | None = None,
    physical: bool = False,
    split: str = "test",
    draw_kind: str = "round",
    run_id: str = RUN,
) -> CellTask:
    cell = CellKey(run_id, seed, split, dec.id, slot, draw_kind, draw, item)
    req = GenRequest(
        system=system if system is not None else f"Prompt for slot {slot}. Put the answer in \\boxed{{}}.",
        user=f"Question {item}: how many apples?",
        decoding=dec,
        seed=req_seed,
        nonce=nonce,
    )
    return CellTask(cell, req, physical)


def proposer_task(round_: int, attempt: int = 0, seed: int = 0) -> CellTask:
    req = GenRequest(
        system=PROPOSER_SYSTEM,
        user=f"<current_prompt>\nPrompt for slot 0.\n</current_prompt> round {round_}",
        decoding=PROPOSER,
        seed=proposer_seed(seed, round_, attempt),
    )
    return CellTask(None, req)


@pytest.fixture(autouse=True)
def _no_env_failure(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(FAIL_AFTER_ENV, raising=False)


@pytest.fixture
def store(tmp_path: Path):
    s = Store(tmp_path / "run" / "driftlab.sqlite")
    yield s
    s.close()


def ledger_sums(store: Store) -> tuple[int, int, int]:
    rows = store.ledger_rows(RUN)
    return (
        sum(r["n_requested"] for r in rows),
        sum(r["n_executed"] for r in rows),
        sum(r["n_cache_hits"] for r in rows),
    )


# ---------------------------------------------------------------- basics
def test_engine_fp_and_key_for(store: Store):
    be = FakeBackend()
    eng = GenerationEngine(be, store, RUN)
    assert eng.engine_fp == engine_fingerprint(be.engine_info())
    assert GenerationEngine(be, store, RUN, engine_fp="pinned").engine_fp == "pinned"
    t = task(0)
    (rec,) = eng.run([t], stage="s", purpose="eval_matrix")
    assert rec.gen_key == eng.key_for(t.request)
    other = GenerationEngine(FakeBackend(engine_tag="e2"), store, RUN)
    assert other.key_for(t.request) != rec.gen_key  # engine fingerprint enters the key
    assert GenerationEngine(FakeBackend(revision="r2"), store, RUN).key_for(t.request) != rec.gen_key


def test_empty_call_writes_nothing(store: Store):
    assert GenerationEngine(FakeBackend(), store, RUN).run([], stage="s", purpose="p") == []
    assert store.count_rows("ledger") == 0


def test_generation_rows_carry_full_provenance(store: Store):
    be = FakeBackend()
    eng = GenerationEngine(be, store, RUN)
    tg, ts = task(1), task(1, dec=T02, req_seed=1234, nonce=None)
    rg, rs = eng.run([tg, ts], stage="matrix", purpose="eval_matrix", seed=0, round_=3)
    rows = store.lookup_generations([rg.gen_key, rs.gen_key])
    g, s = rows[rg.gen_key], rows[rs.gen_key]
    assert g["seed"] is None and s["seed"] == 1234  # seed stored only for sampling
    assert json.loads(g["decoding_json"]) == {"id": "greedy", **GREEDY.params()}
    assert json.loads(s["decoding_json"])["temperature"] == 0.2
    assert g["system_hash"] == sha256_text(tg.request.system)
    assert g["user_hash"] == sha256_text(tg.request.user)
    assert g["rendered_sha"] == sha256_text(be.render(tg.request.system, tg.request.user))
    assert g["engine_fp"] == eng.engine_fp and g["model_id"] == "fake/model" and g["model_revision"] == "r1"
    assert g["batch_id"] == "matrix:0" and g["created_at"]
    assert g["response"] == rg.text and g["finish_reason"] == rg.finish_reason
    assert g["n_completion_tokens"] == rg.n_completion_tokens
    (led,) = store.ledger_rows(RUN)
    assert (led["seed"], led["stage"], led["purpose"], led["round"]) == (0, "matrix", "eval_matrix", 3)
    assert led["completion_tokens"] == rg.n_completion_tokens + rs.n_completion_tokens
    assert led["prompt_tokens"] == g["n_prompt_tokens"] + s["n_prompt_tokens"]


# ---------------------------------------------------------------- caching
def test_cache_hits_across_calls(store: Store):
    be = FakeBackend()
    eng = GenerationEngine(be, store, RUN, chunk_size=4)
    first = eng.run(
        [task(i, physical=True) for i in range(10)], stage="dev", purpose="trajectory_dev", round_=0
    )
    assert be.n_generated == 10 and be.calls == 3
    assert not any(r.cached for r in first)
    # same prompts at a later round, non-physical: served from the cache, no model calls
    second = eng.run([task(i, draw=1) for i in range(10)], stage="dev", purpose="trajectory_dev", round_=1)
    assert be.n_generated == 10 and be.calls == 3
    assert all(r.cached for r in second)
    assert [r.text for r in second] == [r.text for r in first]
    assert [r.gen_key for r in second] == [r.gen_key for r in first]
    rows = store.ledger_rows(RUN)
    assert [(r["n_requested"], r["n_executed"], r["n_cache_hits"]) for r in rows] == [
        (4, 4, 0),
        (4, 4, 0),
        (2, 2, 0),
        (10, 0, 10),  # all-cache-hit call: one zero-generation ledger row
    ]
    assert ledger_sums(store) == (20, 10, 10)
    assert store.count_rows("cells") == 20 and store.count_rows("generations") == 10
    cells = {(c["draw"], c["item_idx"]): c for c in store.get_cells(RUN)}
    assert cells[(0, 3)]["gen_key"] == cells[(1, 3)]["gen_key"]
    assert cells[(0, 3)]["physical"] == 1 and cells[(1, 3)]["physical"] == 0
    # a new engine instance (e.g. after a restart) sees the same cache
    eng2 = GenerationEngine(FakeBackend(), store, RUN)
    assert all(r.cached for r in eng2.run([task(i, draw=2) for i in range(10)], stage="x", purpose="p"))


def test_dedupe_within_call(store: Store):
    be = FakeBackend()
    eng = GenerationEngine(be, store, RUN, chunk_size=2)
    tasks = [task(0), task(1), task(0, draw=1), task(2), task(1, draw=1), task(0, draw=2)]
    recs = eng.run(tasks, stage="s", purpose="p")
    assert be.n_generated == 3
    assert [r.cached for r in recs] == [False, False, True, False, True, True]
    assert recs[0].gen_key == recs[2].gen_key == recs[5].gen_key and recs[0].text == recs[5].text
    rows = store.ledger_rows(RUN)
    # chunk 0 generates items 0, 1 and commits their 4 cells; chunk 1 generates item 2
    assert [(r["n_requested"], r["n_executed"], r["n_cache_hits"]) for r in rows] == [(5, 2, 3), (1, 1, 0)]
    assert store.count_rows("cells") == 6


def test_physical_nonce_forces_regeneration(store: Store):
    be = FakeBackend()
    eng = GenerationEngine(be, store, RUN)
    (orig,) = eng.run([task(0, physical=True)], stage="s", purpose="p")
    (logical,) = eng.run([task(0, draw=1)], stage="s", purpose="p")
    (rerun,) = eng.run([task(0, draw=2, nonce="rerun:2", physical=True)], stage="s", purpose="p")
    assert be.n_generated == 2
    assert logical.cached and logical.gen_key == orig.gen_key
    assert not rerun.cached and rerun.gen_key != orig.gen_key
    assert be.seen[-1].nonce == "rerun:2"
    assert store.lookup_generations([rerun.gen_key])[rerun.gen_key]["nonce"] == "rerun:2"
    # the same physical rerun requested again is itself cached (idempotent re-runs)
    (again,) = eng.run([task(0, draw=2, nonce="rerun:2", physical=True)], stage="s", purpose="p")
    assert again.cached and again.gen_key == rerun.gen_key and be.n_generated == 2


def test_greedy_ignores_seed_sampling_does_not(store: Store):
    be = FakeBackend()
    eng = GenerationEngine(be, store, RUN)
    g = eng.run([task(0, req_seed=1), task(0, draw=1, req_seed=2), task(0, draw=2)], stage="s", purpose="p")
    assert be.n_generated == 1 and g[1].cached and g[2].cached
    seeds = [sample_seed(0, "eval", "t02", 0, "round", r, "test", 0) for r in range(3)]
    s = eng.run([task(0, draw=r, dec=T02, req_seed=sd) for r, sd in enumerate(seeds)], stage="s", purpose="p")
    assert be.n_generated == 4 and len({r.gen_key for r in s}) == 3 and not any(r.cached for r in s)
    # lean mode: every round re-uses one sample seed -> cache hits
    lean = sample_seed(0, "eval", "t02", 0, "round", -1, "test", 0)
    s2 = eng.run(
        [task(0, draw=r, dec=T02, req_seed=lean, slot=0) for r in range(3, 6)], stage="s", purpose="p"
    )
    assert be.n_generated == 5 and [r.cached for r in s2] == [False, True, True]


def test_proposer_tasks_are_stored_without_cells(store: Store):
    be = FakeBackend()
    eng = GenerationEngine(be, store, RUN)
    recs = eng.run(
        [proposer_task(1), proposer_task(1, attempt=1), task(0)], stage="trajectory", purpose="proposer"
    )
    assert be.n_generated == 3
    assert store.count_rows("generations") == 3 and store.count_rows("cells") == 1
    gens = store.lookup_generations([r.gen_key for r in recs[:2]])
    assert {g["seed"] for g in gens.values()} == {proposer_seed(0, 1, 0), proposer_seed(0, 1, 1)}
    assert all(json.loads(g["decoding_json"])["id"] == "proposer" for g in gens.values())
    assert ledger_sums(store) == (3, 3, 0)
    # proposer generations are never offered for scoring (no cell -> no gold)
    store.put_items("test", [{"idx": 0, "question": "q", "answer_text": "a", "gold": "1"}])
    assert [u["gen_key"] for u in store.unscored("v1", "h")] == [recs[2].gen_key]


def test_ledger_sums_match_tasks_and_physical_generations(store: Store):
    be = FakeBackend()
    eng = GenerationEngine(be, store, RUN, chunk_size=3)
    eng.run([task(i, physical=True) for i in range(5)], stage="a", purpose="p")
    mixed = (
        [task(i, draw=1) for i in range(5)]  # cache hits
        + [task(i, slot=1, system="Other prompt.") for i in range(7)]  # new
        + [task(i, slot=1, system="Other prompt.", draw=1) for i in range(7)]  # dupes of the new ones
        + [proposer_task(2)]
    )
    recs = eng.run(mixed, stage="b", purpose="p")
    assert len(recs) == len(mixed) and all(isinstance(r, GenRecord) for r in recs)
    assert ledger_sums(store) == (5 + len(mixed), be.n_generated, 5 + len(mixed) - be.n_generated)
    assert be.n_generated == 5 + 7 + 1
    assert eng.n_requested == 5 + len(mixed) and eng.n_executed == be.n_generated


def test_cell_from_other_run_rejected(store: Store):
    eng = GenerationEngine(FakeBackend(), store, RUN)
    with pytest.raises(ValueError):
        eng.run([task(0, run_id="someone_else")], stage="s", purpose="p")
    assert store.count_rows("generations") == 0


def test_on_progress_reports_each_chunk(store: Store):
    events: list[dict] = []
    eng = GenerationEngine(FakeBackend(), store, RUN, chunk_size=2, on_progress=events.append)
    eng.run([task(i) for i in range(5)], stage="m", purpose="eval_matrix", seed=1, round_=2)
    assert [(e["chunk"], e["n_chunks"], e["n_done"]) for e in events] == [(1, 3, 2), (2, 3, 4), (3, 3, 5)]
    assert events[0]["stage"] == "m" and events[0]["seed"] == 1 and events[0]["round"] == 2


def test_error_results_are_not_cached(store: Store):
    bad_user = task(1).request.user
    be = FakeBackend(fail_users={bad_user})
    eng = GenerationEngine(be, store, RUN, chunk_size=10)
    with pytest.raises(GenerationError):
        eng.run([task(i) for i in range(3)], stage="s", purpose="p")
    assert store.count_rows("generations") == 2 and store.count_rows("cells") == 2
    assert ledger_sums(store) == (2, 2, 0)
    fixed = GenerationEngine(FakeBackend(), store, RUN, chunk_size=10)
    recs = fixed.run([task(i) for i in range(3)], stage="s", purpose="p")
    assert [r.cached for r in recs] == [True, False, True]
    assert store.count_rows("cells") == 3


def test_backend_result_count_mismatch_raises(store: Store):
    class Short(FakeBackend):
        def generate(self, reqs):
            return super().generate(reqs)[:-1]

    with pytest.raises(RuntimeError, match="results for"):
        GenerationEngine(Short(), store, RUN).run([task(0), task(1)], stage="s", purpose="p")
    assert store.count_rows("generations") == 0


# ---------------------------------------------------------------- budgets & resume
def test_deadline_raises_after_committing_a_chunk(store: Store):
    be = FakeBackend()
    eng = GenerationEngine(be, store, RUN, chunk_size=2, deadline=time.monotonic() - 1.0)
    with pytest.raises(BudgetExhausted):
        eng.run([task(i) for i in range(6)], stage="s", purpose="p")
    assert be.n_generated == 2 and eng.chunks_committed == 1
    assert store.count_rows("generations") == 2 and store.count_rows("cells") == 2
    assert ledger_sums(store) == (2, 2, 0)
    # cache-only work is free and still completes after the deadline
    recs = eng.run([task(i, draw=1) for i in range(2)], stage="s", purpose="p")
    assert all(r.cached for r in recs)
    with pytest.raises(BudgetExhausted):
        eng.run([task(i) for i in range(6)], stage="s", purpose="p")
    # a fresh budget finishes the job
    eng2 = GenerationEngine(be, store, RUN, chunk_size=2, deadline=time.monotonic() + 3600)
    eng2.run([task(i) for i in range(6)], stage="s", purpose="p")
    assert store.count_rows("generations") == 6 and be.n_generated == 6


def _workload() -> list[tuple[str, str, int | None, list[CellTask]]]:
    """A miniature pipeline: dev runs, proposer calls, eval matrix with physical reruns and t02 draws."""
    calls: list[tuple[str, str, int | None, list[CellTask]]] = []
    n = 7
    calls.append(
        ("trajectory", "trajectory_dev", 0, [task(i, split="train", physical=True) for i in range(n)])
    )
    calls.append(("trajectory", "proposer", 1, [proposer_task(1, a) for a in range(2)]))
    calls.append(
        (
            "trajectory",
            "candidate_dev",
            1,
            [task(i, split="train", slot=1, draw=1, system="Candidate 1.", physical=True) for i in range(n)],
        )
    )
    matrix: list[CellTask] = []
    for k, system in ((0, None), (1, "Candidate 1.")):
        for r in range(k, 3):
            for i in range(n):
                creation = r == k
                matrix.append(
                    task(i, slot=k, draw=r, system=system, physical=creation or r == 2,
                         nonce=None if creation or r != 2 else f"rerun:{r}")
                )  # fmt: skip
                sd = sample_seed(0, "eval", "t02", k, "round", r, "test", i)
                matrix.append(task(i, slot=k, draw=r, system=system, dec=T02, req_seed=sd, physical=True))
    calls.append(("matrix", "eval_matrix", None, matrix))
    gt = [
        task(i, dec=T02, draw_kind="gt", draw=0, req_seed=sample_seed(0, "gt", "t02", 0, "gt", 0, "test", i))
        for i in range(n)
    ]
    calls.append(("matrix", "gt_draw", None, gt))
    return calls


def _run_workload(engine: GenerationEngine) -> None:
    for stage, purpose, round_, tasks in _workload():
        engine.run(tasks, stage=stage, purpose=purpose, seed=0, round_=round_)


def test_injected_failure_then_resume_matches_uninterrupted_run(tmp_path: Path):
    ref_store = Store(tmp_path / "ref" / "db.sqlite")
    ref_backend = FakeBackend()
    _run_workload(GenerationEngine(ref_backend, ref_store, RUN, chunk_size=5))
    want = ref_store.content_digest(RUN)
    total_tasks = sum(len(t) for *_, t in _workload())

    store = Store(tmp_path / "crash" / "db.sqlite")
    shard_dir = tmp_path / "drive" / "shards"
    be = FakeBackend()
    crashed = GenerationEngine(
        be,
        store,
        RUN,
        chunk_size=5,
        fail_after_chunks=3,
        shard_writer=ShardWriter.for_store(store, shard_dir),
    )
    with pytest.raises(InjectedFailure):
        _run_workload(crashed)
    assert crashed.chunks_executed == 3
    partial = store.content_digest(RUN)
    assert partial != want
    # restart: new engine, same tasks, everything already committed is a cache hit
    resumed = GenerationEngine(
        be, store, RUN, chunk_size=5, shard_writer=ShardWriter.for_store(store, shard_dir)
    )
    _run_workload(resumed)
    assert store.content_digest(RUN) == want
    assert be.n_generated == ref_backend.n_generated  # nothing generated twice
    req, executed, _ = ledger_sums(store)
    assert executed == ref_store.count_rows("generations")
    assert req >= total_tasks  # the re-submitted calls are logged as cache hits
    # the shard log alone rebuilds the identical store (Drive restore without any snapshot)
    restored = Store(tmp_path / "restored" / "db.sqlite")
    replay_shards(restored, shard_dir)
    assert restored.content_digest(RUN) == want
    for s in (ref_store, store, restored):
        s.close()


def test_env_var_failures_converge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ref_store = Store(tmp_path / "ref.sqlite")
    _run_workload(GenerationEngine(FakeBackend(), ref_store, RUN, chunk_size=4))
    want = ref_store.content_digest(RUN)

    monkeypatch.setenv(FAIL_AFTER_ENV, "2")
    store = Store(tmp_path / "crashy.sqlite")
    attempts = 0
    while True:
        attempts += 1
        assert attempts < 100, "no progress across restarts"
        engine = GenerationEngine(FakeBackend(), store, RUN, chunk_size=4)
        assert engine.fail_after_chunks == 2
        try:
            _run_workload(engine)
            break
        except InjectedFailure:
            continue
    assert attempts > 2
    assert store.content_digest(RUN) == want
    ref_store.close()
    store.close()


def test_fail_after_zero_chunks_generates_nothing(store: Store):
    eng = GenerationEngine(FakeBackend(), store, RUN, fail_after_chunks=0)
    with pytest.raises(InjectedFailure):
        eng.run([task(0)], stage="s", purpose="p")
    assert store.count_rows("generations") == 0 and store.count_rows("ledger") == 0
