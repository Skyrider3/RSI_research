"""Phase A trajectory: determinism, dev-only proposer inputs, fallbacks, advance rules, ledger, resume."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from driftlab.analysis.cube import Trajectory
from driftlab.backends import make_backend
from driftlab.config import REPO_ROOT, ExperimentConfig, load_config
from driftlab.data import answer_key, load_dev, load_eval
from driftlab.engine import GenerationEngine
from driftlab.extraction import extract, is_correct
from driftlab.keys import engine_fingerprint
from driftlab.prompting import extract_current_prompt, load_prompt_file
from driftlab.store import ShardWriter, Store, list_shards, read_shard, replay_shards
from driftlab.trajectory import (
    TrajectoryIncomplete,
    run_trajectory,
    should_advance,
    trajectories_from_store,
    trajectory_progress,
)

SMOKE = REPO_ROOT / "configs" / "smoke_mock.yaml"


class Run:
    """Store + mock engine for one smoke-sized Phase A run."""

    def __init__(self, tmp: Path, overrides: list[str] | None = None, name: str = "store") -> None:
        self.cfg: ExperimentConfig = load_config(SMOKE, overrides)
        self.run_id = self.cfg.run.name
        self.store = Store(tmp / f"{name}.sqlite")
        self.backend = make_backend(self.cfg, answer_key=answer_key(self.cfg))
        self.store.ensure_run(
            self.run_id,
            config_json=self.cfg.model_dump(mode="json"),
            config_hash=self.cfg.config_hash(),
            engine_fp=engine_fingerprint(self.backend.engine_info()),
            synthetic=True,
        )
        self.dev = load_dev(self.cfg)
        self.checkpoints = 0
        self.logs: list[str] = []

    def engine(self) -> GenerationEngine:
        return GenerationEngine(self.backend, self.store, self.run_id, chunk_size=5000)

    def run(self, engine: GenerationEngine | None = None) -> dict[int, Trajectory]:
        def cp() -> None:
            self.checkpoints += 1

        return run_trajectory(
            self.cfg,
            engine or self.engine(),
            self.store,
            self.run_id,
            self.dev,
            self.cfg.run.seeds,
            log=self.logs.append,
            checkpoint=cp,
        )

    def ledger_executed(self) -> int:
        return sum(r["n_executed"] for r in self.store.ledger_rows(self.run_id))


def _plain(t: Trajectory) -> dict:
    """Comparable fields (NaN -> None)."""
    fix = [None if isinstance(x, float) and math.isnan(x) else x for x in t.inc_dev_acc]
    return {
        "prompts": t.prompts,
        "hashes": t.prompt_hashes,
        "inc_slot": t.inc_slot,
        "advanced": t.advanced,
        "inc_dev_acc": fix,
        "cand_dev_acc": t.cand_dev_acc,
        "is_fallback": t.is_fallback,
        "origin": t.origin,
        "created_round": t.created_round,
    }


@pytest.fixture(scope="module")
def done(tmp_path_factory: pytest.TempPathFactory) -> tuple[Run, dict[int, Trajectory]]:
    r = Run(tmp_path_factory.mktemp("traj"))
    return r, r.run()


# --------------------------------------------------------------------------- structure and determinism


def test_structure(done) -> None:
    r, trajs = done
    R = r.cfg.run.rounds
    assert sorted(trajs) == r.cfg.run.seeds
    initial = load_prompt_file(r.cfg.trajectory.initial_prompt_file)
    for s, t in trajs.items():
        assert t.seed == s and t.R == R and len(t.prompts) == R + 1
        assert t.prompts[0] == initial and t.origin[0] == "initial"
        assert t.inc_slot[0] == t.inc_slot[1] == 0 and t.advanced[0] is False
        assert len(set(t.prompts)) == R + 1  # every candidate differs from every earlier slot
        for k in range(1, R + 1):
            assert t.inc_slot[k] < k
            if k < R:
                assert t.inc_slot[k + 1] == (k if t.advanced[k] else t.inc_slot[k])
        assert math.isnan(t.inc_dev_acc[0])
        assert all(0.0 <= a <= 1.0 for a in t.cand_dev_acc)
        rows = r.store.get_trajectory_rounds(r.run_id, s)
        assert [x["round"] for x in rows] == list(range(1, R + 1))  # rounds 1..R only
        assert rows[0]["inc_dev_acc"] == t.cand_dev_acc[0]  # slot 0's dev accuracy
        slots = r.store.get_slots(r.run_id, s)
        assert [x["prompt_text"] for x in slots] == t.prompts
        assert [x["parent_slot"] for x in slots] == [None] + t.inc_slot[1:]
    assert trajectory_progress(r.store, r.run_id, r.cfg.run.seeds) == {s: R for s in r.cfg.run.seeds}


def test_two_fresh_runs_identical(done, tmp_path: Path) -> None:
    r, trajs = done
    other = Run(tmp_path, name="other")
    trajs2 = other.run()
    assert {s: _plain(t) for s, t in trajs.items()} == {s: _plain(t) for s, t in trajs2.items()}
    strip = lambda rows: [{k: v for k, v in x.items() if k != "completed_at"} for x in rows]  # noqa: E731
    assert strip(r.store.get_trajectory_rounds(r.run_id)) == strip(
        other.store.get_trajectory_rounds(r.run_id)
    )
    assert r.store.get_proposals(r.run_id) == other.store.get_proposals(r.run_id)
    assert r.store.content_digest(r.run_id) == other.store.content_digest(r.run_id)


def test_dev_cells_and_accuracies(done) -> None:
    r, trajs = done
    gold = {it.idx: it.gold for it in r.dev}
    for s, t in trajs.items():
        for k in range(t.R + 1):
            rows = r.store.get_cell_texts(r.run_id, s, "train", "greedy", k, "round", k)
            assert [x["item_idx"] for x in rows] == [it.idx for it in r.dev]
            acc = sum(is_correct(extract("v1", x["response"]), gold[x["item_idx"]]) for x in rows) / len(rows)
            assert acc == pytest.approx(t.cand_dev_acc[k])
        for k in range(1, t.R + 1):
            assert t.inc_dev_acc[k] == pytest.approx(t.cand_dev_acc[t.inc_slot[k]])
    assert not r.store.get_cells(r.run_id, split="test")  # Phase A never touches the eval split


def test_ledger_purposes_split_by_seed(done) -> None:
    r, _ = done
    rows = r.store.ledger_rows(r.run_id)
    assert {x["stage"] for x in rows} == {"trajectory"}
    assert {x["purpose"] for x in rows} == {"trajectory_dev", "candidate_dev", "proposer"}
    assert all(x["seed"] in r.cfg.run.seeds for x in rows)
    dev0 = {x["seed"]: x for x in rows if x["purpose"] == "trajectory_dev"}
    D = len(r.dev)
    assert dev0[0]["n_executed"] == D and dev0[0]["round"] == 0
    assert dev0[1]["n_executed"] == 0 and dev0[1]["n_cache_hits"] == D  # greedy slot 0 shared by the cache
    cand = [x for x in rows if x["purpose"] == "candidate_dev"]
    assert sorted((x["seed"], x["round"]) for x in cand) == [(s, t) for s in (0, 1) for t in range(1, 5)]
    n_prop = sum(x["n_requested"] for x in rows if x["purpose"] == "proposer")
    assert n_prop == len(r.store.get_proposals(r.run_id))


# --------------------------------------------------------------------------- proposer inputs


def test_no_eval_text_in_meta_prompts(done) -> None:
    r, trajs = done
    eval_questions = load_eval(r.cfg).questions()
    dev_by_idx = {it.idx: it for it in r.dev}
    props = r.store.get_proposals(r.run_id)
    assert props
    for p in props:
        meta = r.store.get_prompt(p["meta_prompt_hash"])
        assert meta is not None
        assert not any(q in meta for q in eval_questions)
        idxs = json.loads(p["error_item_idxs"])
        assert len(idxs) <= r.cfg.trajectory.proposer.n_errors
        for i in idxs:
            assert dev_by_idx[i].question in meta
        traj = trajs[p["seed"]]
        inc = traj.inc_slot[p["round"]]
        assert extract_current_prompt(meta) == traj.prompts[inc]
        gen = r.store.lookup_generations([p["gen_key"]])[p["gen_key"]]
        assert gen["seed"] is not None and json.loads(gen["decoding_json"])["id"] == "proposer"


def test_shown_errors_are_the_incumbents_dev_errors(done) -> None:
    """Each meta-prompt shows min(n_errors, #errors) dev items the incumbent got WRONG (re-derived from the stored
    dev texts), with the incumbent's own (truncated) response, and nothing it got right."""
    r, trajs = done
    k_err = r.cfg.trajectory.proposer.n_errors
    head = r.cfg.trajectory.proposer.response_head_chars
    dev = {it.idx: it for it in r.dev}
    for p in r.store.get_proposals(r.run_id):
        traj = trajs[p["seed"]]
        inc = traj.inc_slot[p["round"]]
        rows = r.store.get_cell_texts(r.run_id, p["seed"], "train", "greedy", inc, "round", inc)
        text = {x["item_idx"]: x["response"] for x in rows}
        wrong = {i for i, t in text.items() if not is_correct(extract("v1", t), dev[i].gold)}
        shown = json.loads(p["error_item_idxs"])
        assert set(shown) <= wrong and len(shown) == min(k_err, len(wrong)) == len(set(shown))
        meta = r.store.get_prompt(p["meta_prompt_hash"])
        for i in shown:
            assert text[i][:head] in meta
        for i in set(text) - wrong:
            assert dev[i].question not in meta


def test_proposals_restorable_from_the_shard_log(tmp_path: Path) -> None:
    """Every proposal's meta-prompt and parsed candidate come back from the shards alone, and each attempt is
    ONE shard (prompts + proposal row in one transaction) rather than one shard per row."""
    r = Run(tmp_path, ["run.rounds=2"])
    shards = tmp_path / "shards"
    sw = ShardWriter.for_store(r.store, shards)
    eng = GenerationEngine(r.backend, r.store, r.run_id, chunk_size=5000, shard_writer=sw)
    r.run(eng)
    props = r.store.get_proposals(r.run_id)
    assert props and any(p["parsed_prompt_hash"] for p in props)
    fresh = Store(tmp_path / "fresh.sqlite")
    replay_shards(fresh, shards)
    assert fresh.get_proposals(r.run_id) == props
    for p in props:
        assert (
            fresh.get_prompt(p["meta_prompt_hash"]) == r.store.get_prompt(p["meta_prompt_hash"]) is not None
        )
        if p["parsed_prompt_hash"]:
            assert fresh.get_prompt(p["parsed_prompt_hash"]) is not None
    proposal_shards = [p for _, p in list_shards(shards) if "proposals" in read_shard(p)]
    assert len(proposal_shards) == len(props)
    for path in proposal_shards:
        tables = read_shard(path)
        assert set(tables) == {"prompts", "proposals"} and len(tables["proposals"]) == 1
    fresh.close()


def test_type_error_on_eval_split(tmp_path: Path) -> None:
    r = Run(tmp_path)
    with pytest.raises(TypeError, match="DEV split"):
        run_trajectory(r.cfg, r.engine(), r.store, r.run_id, load_eval(r.cfg), r.cfg.run.seeds)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        run_trajectory(r.cfg, r.engine(), r.store, r.run_id, list(r.dev), r.cfg.run.seeds)  # type: ignore[arg-type]
    assert r.store.count_rows("generations") == 0


def test_malformed_proposals_force_fallbacks(tmp_path: Path) -> None:
    r = Run(tmp_path, ["backend.mock.malformed_proposal_rate=1.0"])
    trajs = r.run()
    m = r.cfg.trajectory.proposer.max_attempts
    for s, t in trajs.items():
        assert t.is_fallback[1:] == [True] * t.R and t.origin[1:] == ["fallback"] * t.R
        assert len(set(t.prompts)) == t.R + 1
        for row in r.store.get_trajectory_rounds(r.run_id, s):
            assert row["n_attempts"] == m and row["is_fallback"] == 1
            parent = t.prompts[row["incumbent_slot"]]
            assert t.prompts[row["round"]].startswith(parent.rstrip())
    props = r.store.get_proposals(r.run_id)
    assert len(props) == len(r.cfg.run.seeds) * r.cfg.run.rounds * m
    assert all(p["valid"] == 0 and p["violations"] != "[]" for p in props)


# --------------------------------------------------------------------------- advance rules


@pytest.mark.parametrize("rule", ["dev_gt", "dev_ge", "always"])
def test_advance_rules(tmp_path: Path, rule: str) -> None:
    r = Run(tmp_path, [f"trajectory.advance_rule={rule}"])
    trajs = r.run()
    D = len(r.dev)
    for t in trajs.values():
        for k in range(1, t.R + 1):
            cand, inc = round(t.cand_dev_acc[k] * D), round(t.inc_dev_acc[k] * D)
            want = {"dev_gt": cand > inc, "dev_ge": cand >= inc, "always": True}[rule]
            assert t.advanced[k] == want
        if rule == "always":
            assert t.inc_slot == [0, 0, 1, 2, 3]


def test_static_mode_never_advances(tmp_path: Path) -> None:
    r = Run(tmp_path, ["trajectory.mode=static", "trajectory.advance_rule=always"])
    trajs = r.run()
    for t in trajs.values():
        assert t.advanced == [False] * (t.R + 1) and t.inc_slot == [0] * (t.R + 1)
        assert len(set(t.prompts)) == t.R + 1
    for p in r.store.get_proposals(r.run_id):  # every candidate is proposed from slot 0
        assert extract_current_prompt(r.store.get_prompt(p["meta_prompt_hash"])) == trajs[0].prompts[0]


def test_should_advance_rules() -> None:
    base = load_config(SMOKE)
    assert should_advance(base, 5, 4) and not should_advance(base, 4, 4)
    ge = load_config(SMOKE, ["trajectory.advance_rule=dev_ge"])
    assert should_advance(ge, 4, 4) and not should_advance(ge, 3, 4)
    static = load_config(SMOKE, ["trajectory.mode=static"])
    assert not should_advance(static, 9, 0)


# --------------------------------------------------------------------------- resume


def test_resume_without_deletions_is_a_noop(tmp_path: Path) -> None:
    r = Run(tmp_path)
    first = r.run()
    n_rows, n_shardless = len(r.store.ledger_rows(r.run_id)), r.store.count_rows("generations")
    cps = r.checkpoints
    assert cps == r.cfg.run.rounds + 1  # round 0 + every round
    eng = r.engine()
    again = r.run(eng)
    assert eng.n_executed == 0 and eng.n_requested == 0  # no engine call at all
    assert len(r.store.ledger_rows(r.run_id)) == n_rows and r.store.count_rows("generations") == n_shardless
    assert r.checkpoints == cps  # nothing committed -> no checkpoint
    assert {s: _plain(t) for s, t in first.items()} == {s: _plain(t) for s, t in again.items()}
    rebuilt = trajectories_from_store(r.cfg, r.store, r.run_id)
    assert {s: _plain(t) for s, t in first.items()} == {s: _plain(t) for s, t in rebuilt.items()}


def test_resume_after_interrupted_rounds(tmp_path: Path) -> None:
    r = Run(tmp_path)
    first = r.run()
    digest = r.store.content_digest(r.run_id)
    executed = r.ledger_executed()
    with r.store.transaction():  # crash before rounds 3..4 were marked complete (seed 1) / round 4 (seed 0)
        r.store.conn.execute("DELETE FROM trajectory_rounds WHERE (seed = 1 AND round >= 3) OR round = 4")
    with pytest.raises(TrajectoryIncomplete):
        trajectories_from_store(r.cfg, r.store, r.run_id)
    again = r.run()
    assert {s: _plain(t) for s, t in first.items()} == {s: _plain(t) for s, t in again.items()}
    assert r.ledger_executed() == executed  # recomputed rounds are cache hits
    assert r.store.content_digest(r.run_id) == digest


def test_sampling_trajectory_env(tmp_path: Path) -> None:
    r = Run(tmp_path, ["trajectory.env=E2", "run.rounds=2"])
    trajs = r.run()
    assert all(t.R == 2 for t in trajs.values())
    dev0 = {x["seed"]: x for x in r.store.ledger_rows(r.run_id) if x["purpose"] == "trajectory_dev"}
    assert dev0[0]["n_executed"] == dev0[1]["n_executed"] == len(r.dev)  # seeded per seed: not shared
    cells = r.store.get_cells(r.run_id, split="train")
    assert {c["decoding_id"] for c in cells} == {"t02"}
