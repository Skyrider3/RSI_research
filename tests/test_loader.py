"""DB -> Cube / Trajectory loader (stores built directly with the engine + MockBackend; no Pipeline).

``build_run`` is also used by tests/test_bundle.py: it writes a complete small run dir (config.yaml, plan.yaml,
store.sqlite) with 2 seeds, R rounds, N eval items, both decodings over the full triangle, physical greedy
reruns at chosen ages, GT draws, audit draws, dev cells, a 'proposer' cell (must be ignored), scores for v1/v2
computed with the frozen extractors, ledger rows (from the engine) and audit_results rows.
"""

from __future__ import annotations

import math
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from driftlab.analysis.cube import Cube
from driftlab.analysis.loader import (
    TrajectoryInconsistent,
    greedy_decodings,
    load_cube,
    load_trajectories,
    proposer_attempts,
    score_status,
)
from driftlab.backends.base import GenRequest
from driftlab.backends.mock import MockBackend
from driftlab.config import REPO_ROOT, ExperimentConfig, dump_yaml, load_config, load_plan
from driftlab.data import answer_key, load_dev, load_eval, user_message
from driftlab.engine import CellKey, CellTask, GenerationEngine
from driftlab.extraction import REGISTRY, extract, extractor_hash, is_correct
from driftlab.keys import proposer_seed, sample_seed, sha256_text
from driftlab.prompting import PROPOSER_SYSTEM
from driftlab.store.store import Store

SMOKE = REPO_ROOT / "configs" / "smoke_mock.yaml"
PLAN = REPO_ROOT / "analysis_plans" / "prereg_v1.yaml"
BASE_PROMPT = (
    "You are a helpful assistant that solves grade-school math word problems. "
    "Solve the problem step by step, and put your final answer within \\boxed{}."
)
EDITS = (
    "Double-check each arithmetic step before answering.",
    "Answer as briefly as possible.",
    "Read the question carefully and identify what is asked.",
    "Use your intuition rather than lengthy calculations.",
    "Keep track of units and convert them when needed.",
    "State units in the final answer.",
)

CellId = tuple[int, str, int, tuple[str, int]]  # (seed, decoding, slot, draw)


@dataclass
class BuiltRun:
    run_dir: Path
    db: Path
    cfg: ExperimentConfig
    run_id: str
    seeds: list[int]
    R: int
    N: int
    prompts: dict[int, list[str]]
    advanced: dict[int, list[bool]]
    rounds_done: dict[int, list[int]]
    n_attempts: dict[int, dict[int, int]]
    # cell -> {"gen_keys", "texts", "finish", "physical", "correct": {extractor: [0/1]}}
    cells: dict[CellId, dict] = field(default_factory=dict)

    def inc_slot(self, seed: int) -> list[int]:
        adv = self.advanced[seed]
        inc = [0] * (self.R + 1)
        for t in range(2, self.R + 1):
            inc[t] = t - 1 if adv[t - 1] else inc[t - 1]
        return inc


def _prompts(seed: int, R: int) -> list[str]:
    out = [BASE_PROMPT]  # slot 0 is the same initial prompt in every seed (shared greedy cache entry)
    for k in range(1, R + 1):
        out.append(f"{out[k - 1]} {EDITS[(seed * 3 + k) % len(EDITS)]}".strip())
    return out


def build_run(
    root: Path,
    *,
    seeds: Sequence[int] = (0, 1),
    R: int = 3,
    N: int = 12,
    physical_ages: Sequence[int] = (1,),
    gt_draws: int = 1,
    audit_repeats: int = 1,
    advanced: dict[int, list[bool]] | None = None,
    rounds_done: dict[int, list[int]] | None = None,
    flip_rate: float = 0.35,
    matrix_rounds: int | None = None,
    score: bool = True,
    overrides: Sequence[str] = (),
) -> BuiltRun:
    """Write a small complete (or partial: ``rounds_done`` / ``matrix_rounds``) run dir under ``root``."""
    seeds = [int(s) for s in seeds]
    cfg = load_config(
        SMOKE,
        overrides=[
            f"run.seeds=[{', '.join(map(str, seeds))}]",
            f"run.rounds={R}",
            f"data.eval.n={N}",
            f"data.dev.n={N}",
            f"matrix.physical_ages=[{', '.join(map(str, physical_ages))}]",
            f"matrix.gt_draws={gt_draws}",
            f"backend.mock.greedy_flip_rate={flip_rate}",
            f"audit.repeats={audit_repeats}",
            f"audit.n_items={N}",
            *overrides,
        ],
    )
    run_id = cfg.run.name
    run_dir = Path(root) / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(dump_yaml(cfg))
    shutil.copyfile(PLAN, run_dir / "plan.yaml")
    plan = load_plan(PLAN)
    db = run_dir / "store.sqlite"
    store = Store(db)
    backend = MockBackend(cfg.model.id, cfg.model.revision, cfg.backend.mock, answer_key=answer_key(cfg))
    engine = GenerationEngine(backend, store, run_id, chunk_size=100_000)
    store.ensure_run(
        run_id,
        config_json=cfg.model_dump(mode="json"),
        config_hash=cfg.config_hash(),
        plan_hash=plan.plan_hash(),
        engine_fp=engine.engine_fp,
        synthetic=False,  # the mock engine must make it sticky-synthetic on its own
    )
    ev, dev = load_eval(cfg), load_dev(cfg)
    store.put_items("test", [it.to_row() for it in ev])
    store.put_items("train", [it.to_row() for it in dev])
    gold = {("test", it.idx): it.gold for it in ev} | {("train", it.idx): it.gold for it in dev}
    users = [user_message(cfg, it.question) for it in ev]
    greedy, t02 = cfg.decoding("greedy"), cfg.decoding("t02")

    advanced = advanced or {s: [False] + [(t + s) % 2 == 1 for t in range(1, R + 1)] for s in seeds}
    rounds_done = rounds_done or {s: list(range(1, R + 1)) for s in seeds}
    built = BuiltRun(run_dir, db, cfg, run_id, seeds, R, N, {}, advanced, rounds_done, {})
    for s in seeds:
        prompts = built.prompts[s] = _prompts(s, R)
        inc = built.inc_slot(s)
        built.n_attempts[s] = {}
        for k in range(R + 1):
            origin = "initial" if k == 0 else ("fallback" if k == 2 else "proposer")
            store.put_slot(run_id, s, k, prompts[k], k, None if k == 0 else inc[k], origin)
        for t in rounds_done[s]:
            built.n_attempts[s][t] = 1 + (t + s) % 3
            store.put_trajectory_round(
                {
                    "run_id": run_id,
                    "seed": s,
                    "round": t,
                    "incumbent_slot": inc[t],
                    "candidate_slot": t,
                    "inc_dev_acc": 0.5 + 0.01 * inc[t],
                    "cand_dev_acc": 0.5 + 0.01 * t,
                    "advanced": int(advanced[s][t]),
                    "n_attempts": built.n_attempts[s][t],
                    "is_fallback": int(t == 2),
                }
            )

    def add(
        seed: int, dec, slot: int, draw: tuple[str, int], physical: bool, nonce=None, seed_fn=None
    ) -> list:
        cid = (seed, dec.id, slot, draw)
        tasks = []
        for n, it in enumerate(ev):
            rs = None if dec.is_greedy else seed_fn(it.idx)
            tasks.append(
                CellTask(
                    CellKey(run_id, seed, "test", dec.id, slot, draw[0], draw[1], it.idx),
                    GenRequest(prompts_of[seed][slot], users[n], dec, seed=rs, nonce=nonce),
                    physical,
                )
            )
        built.cells[cid] = {"physical": physical}
        return tasks

    prompts_of = built.prompts
    mr = R if matrix_rounds is None else matrix_rounds
    for s in seeds:
        tasks: list[CellTask] = []
        for r in range(mr + 1):
            for k in range(r + 1):
                nonced = k < r and (r - k) in physical_ages
                tasks += add(
                    s, greedy, k, ("round", r), k == r or nonced, f"rerun:s{s}:{r}" if nonced else None
                )
                tasks += add(
                    s,
                    t02,
                    k,
                    ("round", r),
                    True,
                    seed_fn=lambda idx, s=s, k=k, r=r: sample_seed(
                        s, "eval", "t02", k, "round", r, "test", idx
                    ),
                )
        for g in range(gt_draws):
            for k in range(mr + 1):
                tasks += add(
                    s,
                    t02,
                    k,
                    ("gt", g),
                    True,
                    seed_fn=lambda idx, s=s, k=k, g=g: sample_seed(s, "gt", "t02", k, "gt", g, "test", idx),
                )
        if s == seeds[0]:
            last_inc = built.inc_slot(s)[min(mr, R)]
            for a in range(audit_repeats):
                for k in sorted({0, last_inc}):
                    if k > mr:
                        continue
                    tasks += add(s, greedy, k, ("audit", a), True, f"audit:s{s}:{a}")
                    tasks += add(
                        s,
                        t02,
                        k,
                        ("audit", a),
                        True,
                        f"audit:s{s}:{a}",
                        seed_fn=lambda idx, s=s, k=k: sample_seed(
                            s, "eval", "t02", k, "round", k, "test", idx
                        ),
                    )
        recs = engine.run(tasks, stage="matrix", purpose="eval_matrix", seed=s)
        for t, rec in zip(tasks, recs, strict=True):
            c = t.cell
            e = built.cells[(c.seed, c.decoding_id, c.slot, (c.draw_kind, c.draw))]
            e.setdefault("gen_keys", []).append(rec.gen_key)
            e.setdefault("texts", []).append(rec.text)
            e.setdefault("finish", []).append(rec.finish_reason)
    # cells the cube must ignore: dev (train) cells and a test-split 'proposer' cell
    dev_user = user_message(cfg, dev[0].question)
    engine.run(
        [
            CellTask(
                CellKey(run_id, seeds[0], "train", "greedy", 0, "round", 0, dev[0].idx),
                GenRequest(BASE_PROMPT, dev_user, greedy),
                True,
            ),
            CellTask(
                CellKey(run_id, seeds[0], "test", "proposer", 0, "round", 0, ev[0].idx),
                GenRequest(PROPOSER_SYSTEM, users[0], cfg.proposer_decoding(), seed=proposer_seed(0, 1, 0)),
                True,
            ),
        ],
        stage="trajectory",
        purpose="trajectory_dev",
        seed=seeds[0],
    )
    if score:
        _score(store, gold)
    for e in built.cells.values():
        e["correct"] = {
            x: [
                int(is_correct(extract(x, txt), gold[("test", ev[n].idx)]))
                for n, txt in enumerate(e["texts"])
            ]
            for x in REGISTRY
        }
    store.put_audit_results(
        [
            {
                "run_id": run_id,
                "decoding_id": d,
                "slot": 0,
                "repeat": a,
                "pct_text_identical": 100.0 if d == "greedy" else 90.0,
                "pct_correct_flip": 0.0 if d == "greedy" else 5.0,
                "n": N,
            }
            for d in ("greedy", "t02")
            for a in range(audit_repeats)
        ]
    )
    store.close()
    return built


def _score(store: Store, gold: dict[tuple[str, int], str]) -> None:
    """Score every generation referenced by a cell under every registered extractor (frozen hashes)."""
    rows = store.query(
        "SELECT DISTINCT c.gen_key AS gen_key, c.split AS split, c.item_idx AS item_idx, g.response AS response "
        "FROM cells c JOIN generations g ON g.gen_key = c.gen_key"
    )
    out = []
    for r in rows:
        gd = gold[(r["split"], int(r["item_idx"]))]
        for x in REGISTRY:
            ex = extract(x, r["response"])
            out.append(
                {
                    "gen_key": r["gen_key"],
                    "extractor": x,
                    "ext_hash": extractor_hash(x),
                    "extracted": ex.extracted,
                    "method": ex.method,
                    "gold": gd,
                    "correct": int(is_correct(ex, gd)),
                }
            )
    store.put_scores(out)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> BuiltRun:
    return build_run(tmp_path_factory.mktemp("loader"), audit_repeats=2)


@pytest.fixture(scope="module")
def cube(built: BuiltRun) -> Cube:
    return load_cube(built.db)


# --------------------------------------------------------------------------- cube


def test_axes(cube: Cube, built: BuiltRun):
    R = built.R
    assert cube.run_id == built.run_id
    assert cube.seeds == [0, 1]
    assert cube.decodings == ["greedy", "t02"]
    assert cube.extractors == ["v1", "v2"]
    assert cube.n_slots == R + 1 and cube.R == R
    assert cube.n_items == built.N
    assert cube.draws == [("round", r) for r in range(R + 1)] + [("gt", 0), ("audit", 0), ("audit", 1)]
    assert cube.greedy_decodings == frozenset({"greedy"})
    assert cube.synthetic is True  # sticky flag set by the mock engine even though ensure_run said False
    assert cube.correct.shape == (2, 2, R + 1, R + 4, 2, built.N)


def test_every_cell_matches_the_store(cube: Cube, built: BuiltRun):
    planned = set()
    for (s, d, k, draw), e in built.cells.items():
        si, di, ki, ri = cube._idx(s, d, k, draw)
        planned.add((si, di, ki, ri))
        for xi, x in enumerate(cube.extractors):
            np.testing.assert_array_equal(
                cube.correct[si, di, ki, ri, xi], e["correct"][x], err_msg=str((s, d, k, draw, x))
            )
        keys = [cube.gen_keys[g] for g in cube.gen_row[si, di, ki, ri]]
        assert keys == e["gen_keys"]
        assert bool(cube.physical[si, di, ki, ri]) == e["physical"]
        np.testing.assert_array_equal(cube.truncated[si, di, ki, ri], [f == "length" for f in e["finish"]])
        assert cube.has(s, d, k, draw)
        np.testing.assert_array_equal(
            cube.vec(s, d, k, draw, "v2"), np.asarray(e["correct"]["v2"], dtype=bool)
        )
    # every other cell is missing (upper triangle, greedy GT, seed 1 audit, proposer/dev cells)
    S, D, K, Rr = cube.correct.shape[:4]
    for idx in np.ndindex(S, D, K, Rr):
        if idx not in planned:
            assert (cube.correct[idx] == -1).all() and (cube.gen_row[idx] == -1).all()
            assert not cube.physical[idx] and not cube.truncated[idx].any()
            assert (cube.text_row[idx] == -1).all()
    assert not cube.has(0, "greedy", 2, ("round", 1))  # above the triangle
    assert not cube.has(0, "greedy", 0, ("gt", 0))  # greedy has no GT draws
    assert not cube.has(1, "greedy", 0, ("audit", 0))  # only seed 0 is audited


def test_gen_rows_share_cache_hits_and_text_rows_identify_texts(cube: Cube, built: BuiltRun):
    # non-physical greedy rerun (age 2) = the creation generation; physical (age 1) and t02 reruns are fresh
    assert cube.same_gen_frac((0, "greedy", 0, ("round", 2)), (0, "greedy", 0, ("round", 0))) == 1.0
    assert cube.same_gen_frac((0, "greedy", 0, ("round", 1)), (0, "greedy", 0, ("round", 0))) == 0.0
    assert cube.same_gen_frac((0, "t02", 1, ("round", 2)), (0, "t02", 1, ("round", 1))) == 0.0
    # slot 0 is the same prompt in every seed: its greedy creation cell is one shared cached generation
    assert cube.same_gen_frac((0, "greedy", 0, ("round", 0)), (1, "greedy", 0, ("round", 0))) == 1.0
    # text ids: equal id <=> byte-identical text, over every loaded cell item
    by_text: dict[str, set[int]] = {}
    by_id: dict[int, set[str]] = {}
    for (s, d, k, draw), e in built.cells.items():
        si, di, ki, ri = cube._idx(s, d, k, draw)
        for tid, txt in zip(cube.text_row[si, di, ki, ri], e["texts"], strict=True):
            assert tid >= 0
            by_text.setdefault(txt, set()).add(int(tid))
            by_id.setdefault(int(tid), set()).add(txt)
    assert all(len(v) == 1 for v in by_text.values()) and all(len(v) == 1 for v in by_id.values())
    # flip_rate 0.35: some physical greedy reruns differ, and the cache-hit rerun is byte-identical
    same = cube.same_text_frac((0, "greedy", 0, ("round", 1)), (0, "greedy", 0, ("round", 0)))
    assert 0.0 <= same < 1.0
    assert cube.same_text_frac((0, "greedy", 0, ("round", 2)), (0, "greedy", 0, ("round", 0))) == 1.0
    assert (
        len(cube.gen_keys)
        == len(set(cube.gen_keys))
        == len({g for e in built.cells.values() for g in e["gen_keys"]})
    )


def test_text_ids_without_window_functions_match_the_sql_path(
    cube: Cube, built: BuiltRun, monkeypatch: pytest.MonkeyPatch
):
    """SQLite < 3.25 has no DENSE_RANK: the sha256 fallback must give the same text equivalence classes (ids may
    be numbered differently) and leave every other array unchanged."""
    import driftlab.analysis.loader as loader_mod

    real = loader_mod._cube_sql

    def no_window(n_extractors: int, window: bool) -> str:
        return "SELECT no_such_window_function()" if window else real(n_extractors, window)

    monkeypatch.setattr(loader_mod, "_cube_sql", no_window)
    fb = load_cube(built.db)
    np.testing.assert_array_equal(fb.correct, cube.correct)
    np.testing.assert_array_equal(fb.gen_row, cube.gen_row)
    np.testing.assert_array_equal(fb.physical, cube.physical)
    assert fb.gen_keys == cube.gen_keys
    a, b = cube.text_row.reshape(-1), fb.text_row.reshape(-1)
    np.testing.assert_array_equal(a < 0, b < 0)
    pairs = set(zip(a[a >= 0].tolist(), b[b >= 0].tolist(), strict=True))
    assert len(pairs) == len({x for x, _ in pairs}) == len({y for _, y in pairs})  # a bijection of ids


def test_greedy_decodings_from_config():
    assert greedy_decodings({"decodings": {"greedy": {"temperature": 0.0}, "t02": {"temperature": 0.2}}}) == {
        "greedy"
    }
    assert greedy_decodings({"decodings": {"g0": {"temperature": 0}, "hot": {"temperature": 1.0}}}) == {"g0"}
    assert greedy_decodings({}) == {"greedy"}


def test_hand_written_cells_missing_scores_and_truncation(tmp_path: Path):
    """A hand-written store: unscored items are -1 for that extractor only; 'length' marks truncation; a
    non-synthetic run stays non-synthetic; decodings follow the config's temperatures."""
    db = tmp_path / "s.sqlite"
    st = Store(db)
    st.ensure_run(
        "r",
        config_json={"decodings": {"aaa_hot": {"temperature": 0.7}, "zzz_cold": {"temperature": 0.0}}},
        config_hash="h",
        synthetic=False,
    )
    st.put_items(
        "test",
        [{"idx": i, "question": f"q{i}", "answer_text": f"#### {i}", "gold": str(i)} for i in range(3)],
    )
    gens, cells, scores = [], [], []
    for d in ("aaa_hot", "zzz_cold"):
        for n in range(3):
            key = f"{d}-{n}"
            text = f"so the answer is \\boxed{{{n if n != 1 else 99}}}"
            gens.append(
                {
                    "gen_key": key,
                    "engine_fp": "e",
                    "model_id": "m",
                    "model_revision": "r",
                    "rendered_sha": "x",
                    "system_hash": "s",
                    "user_hash": "u",
                    "decoding_json": "{}",
                    "response": text,
                    "finish_reason": "length" if n == 2 else "stop",
                }
            )
            cells.append(
                {
                    "run_id": "r",
                    "seed": 5,
                    "split": "test",
                    "decoding_id": d,
                    "slot": 0,
                    "draw_kind": "round",
                    "draw": 0,
                    "item_idx": n,
                    "gen_key": key,
                    "physical": 1,
                }
            )
            if not (d == "zzz_cold" and n == 0):  # one item unscored under v2
                scores.append(
                    {
                        "gen_key": key,
                        "extractor": "v2",
                        "ext_hash": "old",
                        "method": "boxed",
                        "gold": str(n),
                        "correct": int(n != 1),
                    }
                )
            scores.append(
                {
                    "gen_key": key,
                    "extractor": "v1",
                    "ext_hash": extractor_hash("v1"),
                    "method": "boxed",
                    "gold": str(n),
                    "correct": int(n != 1),
                }
            )
    # a score row under a different gold (e.g. from another item set) must never reach the cube
    scores.append(
        {
            "gen_key": "zzz_cold-1",
            "extractor": "v1",
            "ext_hash": extractor_hash("v1"),
            "method": "boxed",
            "gold": "99",
            "correct": 1,
        }
    )
    st.write_tables({"generations": gens, "cells": cells, "scores": scores})
    st.close()
    cube = load_cube(db)
    assert cube.decodings == ["zzz_cold", "aaa_hot"]  # greedy-like first
    assert cube.greedy_decodings == frozenset({"zzz_cold"})
    assert cube.synthetic is False and cube.seeds == [5] and cube.draws == [("round", 0)]
    s, d, k, r = cube._idx(5, "zzz_cold", 0, ("round", 0))
    np.testing.assert_array_equal(cube.correct[s, d, k, r, 0], [1, 0, 1])  # v1
    np.testing.assert_array_equal(cube.correct[s, d, k, r, 1], [-1, 0, 1])  # v2: item 0 unscored
    np.testing.assert_array_equal(cube.truncated[s, d, k, r], [False, False, True])
    # distinct generations, two of which share a text across decodings -> same text id
    t_cold = cube.text_row[s, d, k, r]
    t_hot = cube.text_row[cube._idx(5, "aaa_hot", 0, ("round", 0))]
    np.testing.assert_array_equal(t_cold, t_hot)
    assert len(set(t_cold.tolist())) == 3
    stale = score_status(db)
    assert set(stale.loc[stale["stale"].astype(bool), "extractor"]) == {"v2"}


def test_empty_run_gives_an_empty_cube(tmp_path: Path):
    db = tmp_path / "e.sqlite"
    st = Store(db)
    st.ensure_run("r", config_json={}, config_hash="h", synthetic=True)
    st.put_items("test", [{"idx": 0, "question": "q", "answer_text": "#### 1", "gold": "1"}])
    st.close()
    cube = load_cube(db)
    assert cube.seeds == [] and cube.n_slots == 0 and cube.n_items == 1 and cube.synthetic
    assert cube.correct.size == 0 and load_trajectories(db) == {}


def test_unknown_run_raises(built: BuiltRun):
    with pytest.raises(LookupError):
        load_cube(built.db, run_id="nope")


def test_load_cube_does_not_write(built: BuiltRun):
    before = built.db.stat().st_mtime_ns
    load_cube(built.db)
    assert built.db.stat().st_mtime_ns == before


# --------------------------------------------------------------------------- trajectories


def test_trajectories_reconstructed(built: BuiltRun):
    trajs = load_trajectories(built.db)
    assert sorted(trajs) == built.seeds
    for s in built.seeds:
        tr = trajs[s]
        assert tr.seed == s and tr.R == built.R
        assert tr.prompts == built.prompts[s]
        assert tr.prompt_hashes == [sha256_text(p) for p in built.prompts[s]]
        assert tr.created_round == list(range(built.R + 1))
        assert tr.advanced == built.advanced[s]
        assert tr.inc_slot == built.inc_slot(s)
        assert tr.inc_slot[0] == tr.inc_slot[1] == 0
        for t in range(1, built.R):
            assert tr.inc_slot[t + 1] == (t if tr.advanced[t] else tr.inc_slot[t])
        assert math.isnan(tr.inc_dev_acc[0])
        assert tr.inc_dev_acc[1:] == pytest.approx(
            [0.5 + 0.01 * tr.inc_slot[t] for t in range(1, built.R + 1)]
        )
        assert tr.cand_dev_acc[0] == pytest.approx(
            0.5
        )  # slot 0's dev accuracy = round 1's incumbent accuracy
        assert tr.cand_dev_acc[1:] == pytest.approx([0.5 + 0.01 * t for t in range(1, built.R + 1)])
        assert tr.is_fallback == [t == 2 for t in range(built.R + 1)]
        assert tr.origin == ["initial", "proposer", "fallback", "proposer"][: built.R + 1]


def test_proposer_attempts(built: BuiltRun):
    assert proposer_attempts(built.db) == built.n_attempts


def test_partial_trajectory_and_attempts_of_interrupted_round(tmp_path: Path):
    b = build_run(tmp_path, R=3, N=4, rounds_done={0: [1, 2], 1: [1]}, matrix_rounds=1, score=False)
    st = Store(b.db)
    for a in range(2):
        st.put_proposal(
            {
                "run_id": b.run_id,
                "seed": 1,
                "round": 2,
                "attempt": a,
                "meta_prompt_hash": "m",
                "error_item_idxs": [1],
                "valid": 0,
            }
        )
    st.close()
    trajs = load_trajectories(b.db)
    assert trajs[0].R == 2 and trajs[1].R == 1  # slots written for unfinished rounds are ignored
    assert trajs[0].prompts == b.prompts[0][:3]
    att = proposer_attempts(b.db)
    assert att[1] == {1: b.n_attempts[1][1], 2: 2}
    cube = load_cube(b.db)
    assert cube.n_slots == 2 and cube.draws[:2] == [("round", 0), ("round", 1)]
    assert (cube.correct == -1).all()  # nothing scored yet
    assert cube.extractors == []


def test_inconsistent_trajectories_raise(tmp_path: Path):
    b = build_run(tmp_path / "a", R=3, N=2, matrix_rounds=0, score=False)
    st = Store(b.db)
    st.put_trajectory_round(
        {
            "run_id": b.run_id,
            "seed": 0,
            "round": 3,
            "incumbent_slot": 0 if b.inc_slot(0)[3] else 1,
            "candidate_slot": 3,
            "advanced": 0,
            "n_attempts": 1,
            "is_fallback": 0,
        }
    )
    st.close()
    with pytest.raises(TrajectoryInconsistent, match="round 3"):
        load_trajectories(b.db)
    gap = build_run(
        tmp_path / "b", R=3, N=2, rounds_done={0: [1, 3], 1: [1, 2, 3]}, matrix_rounds=0, score=False
    )
    with pytest.raises(TrajectoryInconsistent, match="not complete"):
        load_trajectories(gap.db)
