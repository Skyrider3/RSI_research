"""Phase A: the dev-split prompt trajectory of every seed (docs/ARCHITECTURE.md section 3).

Round 0: slot 0 = the packaged initial prompt, evaluated on the DEV split under the trajectory environment
(``cfg.trajectory.env``: its decoding and extractor). Dev cells are ``(seed, "train", decoding, slot,
("round", slot), item)``; under greedy decoding slot 0's generations are shared across seeds by the cache.

Round t = 1..R, all seeds in lockstep (every seed finishes round t before any seed starts round t + 1):

1. errors = dev items the incumbent answers incorrectly (re-extracted in memory from the stored texts);
2. ``sample_errors`` -> ``build_meta_prompt`` (stored in ``prompts`` so it is restorable);
3. proposer attempts ``0..max_attempts-1`` (seed ``proposer_seed(seed, t, attempt)``), each stored as one
   ``proposals`` row together with its meta-prompt and parsed candidate texts (one transaction, one shard);
   the first valid candidate wins, else ``fallback_candidate`` (origin ``fallback``);
4. ``put_slot`` for the candidate, candidate dev run, advance per ``advance_rule`` (``mode: static`` never
   advances);
5. ``put_trajectory_round`` LAST: it marks the round complete.

Resume: rounds with a ``trajectory_rounds`` row are rebuilt from the store (slots, incumbent, accuracies) without
model calls; dev correctness is re-extracted from the stored cell texts only when a later round needs it. An
interrupted round is simply recomputed: every step is deterministic, so proposer calls and dev runs are cache
hits and the store rows converge. Ledger purposes: ``trajectory_dev`` (slot 0 / incumbent), ``candidate_dev``,
``proposer``; ``engine.run`` is called once per seed per purpose, so the ledger splits by seed.

``trajectory_rounds`` holds rounds 1..R only; slot 0's dev accuracy is round 1's ``inc_dev_acc``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from driftlab import keys
from driftlab.analysis.cube import DRAW_ROUND, Trajectory
from driftlab.backends.base import GenRequest
from driftlab.config import ExperimentConfig
from driftlab.data import DEV_SPLIT, DevSplit, user_message
from driftlab.engine import CellKey, CellTask, GenerationEngine
from driftlab.extraction import extract, is_correct
from driftlab.keys import sha256_text
from driftlab.prompting import load_prompt_file
from driftlab.proposer import (
    DevError,
    build_meta_prompt,
    fallback_candidate,
    parse_candidate,
    proposer_request,
    sample_errors,
    shown_golds,
    validate_candidate,
)
from driftlab.store.store import SlotConflict, Store

STAGE = "trajectory"
PURPOSE_DEV = "trajectory_dev"
PURPOSE_CANDIDATE = "candidate_dev"
PURPOSE_PROPOSER = "proposer"


class TrajectoryIncomplete(RuntimeError):
    """Some (seed, round) of Phase A has no committed ``trajectory_rounds`` row yet."""


def should_advance(cfg: ExperimentConfig, cand_correct: int, inc_correct: int) -> bool:
    """Advance decision from dev correct counts (``mode: static`` never advances)."""
    t = cfg.trajectory
    if t.mode == "static":
        return False
    if t.advance_rule == "dev_gt":
        return cand_correct > inc_correct
    if t.advance_rule == "dev_ge":
        return cand_correct >= inc_correct
    if t.advance_rule == "always":
        return True
    raise ValueError(f"unknown advance_rule {t.advance_rule!r}")


def dev_tasks(
    cfg: ExperimentConfig, run_id: str, seed: int, slot: int, prompt: str, dev: DevSplit
) -> list[CellTask]:
    """Creation-cell dev requests of one slot under the trajectory environment."""
    dec = cfg.environment(cfg.trajectory.env).decoding
    out = []
    for it in dev:
        req_seed = (
            None
            if dec.is_greedy
            else keys.sample_seed(seed, "dev", dec.id, slot, DRAW_ROUND, slot, DEV_SPLIT, it.idx)
        )
        out.append(
            CellTask(
                CellKey(run_id, seed, DEV_SPLIT, dec.id, slot, DRAW_ROUND, slot, it.idx),
                GenRequest(prompt, user_message(cfg, it.question), dec, seed=req_seed),
                True,
            )
        )
    return out


@dataclass
class DevResult:
    """Dev texts and correctness (trajectory extractor) of one slot, in dev-item order."""

    texts: list[str]
    correct: list[bool]

    @property
    def n_correct(self) -> int:
        return sum(self.correct)

    @property
    def acc(self) -> float:
        return self.n_correct / len(self.correct) if self.correct else math.nan


@dataclass
class _SeedState:
    seed: int
    prompts: list[str]
    origin: list[str] = field(default_factory=lambda: ["initial"])
    inc_slot: list[int] = field(default_factory=lambda: [0])
    advanced: list[bool] = field(default_factory=lambda: [False])
    inc_dev_acc: list[float] = field(default_factory=lambda: [math.nan])
    cand_dev_acc: list[float] = field(default_factory=lambda: [math.nan])
    is_fallback: list[bool] = field(default_factory=lambda: [False])

    @property
    def next_round(self) -> int:
        return len(self.prompts)

    @property
    def incumbent(self) -> int:
        """Trajectory incumbent for the next round (after the last decided round)."""
        t = len(self.prompts) - 1
        return t if (t >= 1 and self.advanced[t]) else self.inc_slot[t]

    def push(
        self, prompt: str, origin: str, inc: int, advanced: bool, inc_acc: float, cand_acc: float, fb: bool
    ) -> None:
        self.prompts.append(prompt)
        self.origin.append(origin)
        self.inc_slot.append(inc)
        self.advanced.append(bool(advanced))
        self.inc_dev_acc.append(inc_acc)
        self.cand_dev_acc.append(cand_acc)
        self.is_fallback.append(bool(fb))

    def to_trajectory(self, acc0: float) -> Trajectory:
        return Trajectory(
            seed=self.seed,
            prompts=list(self.prompts),
            prompt_hashes=[sha256_text(p) for p in self.prompts],
            created_round=list(range(len(self.prompts))),
            inc_slot=list(self.inc_slot),
            advanced=list(self.advanced),
            inc_dev_acc=list(self.inc_dev_acc),
            cand_dev_acc=[acc0, *self.cand_dev_acc[1:]],
            is_fallback=list(self.is_fallback),
            origin=list(self.origin),
        )


def _float(v: object) -> float:
    return math.nan if v is None else float(v)  # type: ignore[arg-type]


class _PhaseA:
    def __init__(
        self,
        cfg: ExperimentConfig,
        engine: GenerationEngine | None,
        store: Store,
        run_id: str,
        dev: DevSplit | None,
        log: Callable[[str], None],
    ) -> None:
        self.cfg, self.engine, self.store, self.run_id = cfg, engine, store, run_id
        self.dev, self.log = dev, log
        env = cfg.environment(cfg.trajectory.env)
        self.dec, self.extractor = env.decoding, env.extractor
        self.sw = engine.shard_writer if engine is not None else None
        self.cache: dict[tuple[int, int], DevResult] = {}

    # ------------------------------------------------------------------ dev runs
    def _stored_texts(self, seed: int, slot: int, tasks: Sequence[CellTask]) -> list[str] | None:
        """Texts of an already complete dev cell whose gen_keys match the current requests, else None."""
        assert self.engine is not None
        rows = self.store.get_cell_texts(self.run_id, seed, DEV_SPLIT, self.dec.id, slot, DRAW_ROUND, slot)
        by_idx = {r["item_idx"]: r for r in rows}
        out = []
        for t in tasks:
            assert t.cell is not None
            row = by_idx.get(t.cell.item_idx)
            if row is None or row["gen_key"] != self.engine.key_for(t.request):
                return None
            out.append(row["response"])
        return out

    def dev_result(self, seed: int, slot: int, prompt: str, purpose: str, round_: int) -> DevResult:
        key = (seed, slot)
        if key in self.cache:
            return self.cache[key]
        assert self.engine is not None and self.dev is not None
        tasks = dev_tasks(self.cfg, self.run_id, seed, slot, prompt, self.dev)
        texts = self._stored_texts(seed, slot, tasks)
        if texts is None:
            recs = self.engine.run(tasks, stage=STAGE, purpose=purpose, seed=seed, round_=round_)
            texts = [r.text for r in recs]
        res = DevResult(
            list(texts),
            [is_correct(extract(self.extractor, x), it.gold) for x, it in zip(texts, self.dev, strict=True)],
        )
        self.cache[key] = res
        return res

    # ------------------------------------------------------------------ rounds
    def check_slot0(self, seed: int, slot_rows: Mapping[int, dict], initial: str | None) -> str:
        """Slot 0's prompt (written if missing and ``initial`` is given); SlotConflict on a mismatch."""
        row = slot_rows.get(0)
        if row is None:
            if initial is None:
                raise TrajectoryIncomplete(f"run {self.run_id!r}: seed {seed} has no slot 0")
            self.store.put_slot(
                self.run_id, seed, 0, initial, 0, None, "initial", shard_writer=self.sw, stage=STAGE
            )
            return initial
        if initial is not None and row["prompt_hash"] != sha256_text(initial):
            raise SlotConflict(
                f"run {self.run_id!r} seed {seed}: stored slot 0 differs from {self.cfg.trajectory.initial_prompt_file}"
            )
        return row["prompt_text"]

    def restore_round(self, st: _SeedState, t: int, row: Mapping, slot_rows: Mapping[int, dict]) -> None:
        slot = slot_rows.get(t)
        if slot is None or slot.get("prompt_text") is None:
            raise RuntimeError(
                f"run {self.run_id!r}: seed {st.seed} round {t} is complete but slot {t} is missing"
            )
        inc = st.incumbent
        if int(row["incumbent_slot"]) != inc or int(row["candidate_slot"]) != t:
            raise RuntimeError(
                f"run {self.run_id!r}: seed {st.seed} round {t} stores incumbent {row['incumbent_slot']} / "
                f"candidate {row['candidate_slot']}, the trajectory implies {inc} / {t}"
            )
        st.push(
            slot["prompt_text"],
            slot["origin"],
            inc,
            bool(row["advanced"]),
            _float(row["inc_dev_acc"]),
            _float(row["cand_dev_acc"]),
            bool(row["is_fallback"]),
        )

    def compute_round(self, st: _SeedState, t: int) -> None:
        assert self.engine is not None and self.dev is not None
        cfg, pcfg, seed = self.cfg, self.cfg.trajectory.proposer, st.seed
        inc = st.incumbent
        inc_res = self.dev_result(seed, inc, st.prompts[inc], PURPOSE_DEV, t)
        errors = [
            DevError.from_item(it, text)
            for it, text, ok in zip(self.dev, inc_res.texts, inc_res.correct, strict=True)
            if not ok
        ]
        shown = sample_errors(errors, pcfg.n_errors, seed, t)
        meta = build_meta_prompt(st.prompts[inc], shown, pcfg)
        meta_hash = sha256_text(meta)
        golds = shown_golds(shown)
        prior = list(st.prompts)
        candidate: str | None = None
        n_attempts = 0
        for attempt in range(pcfg.max_attempts):
            n_attempts = attempt + 1
            req = proposer_request(meta, cfg.proposer_decoding(), keys.proposer_seed(seed, t, attempt))
            (rec,) = self.engine.run(
                [CellTask(None, req)], stage=STAGE, purpose=PURPOSE_PROPOSER, seed=seed, round_=t
            )
            parsed = parse_candidate(rec.text)
            violations = validate_candidate(parsed, prior, golds, pcfg)
            prompts = [{"prompt_hash": meta_hash, "text": meta}]
            if parsed is not None:
                prompts.append({"prompt_hash": sha256_text(parsed), "text": parsed})
            # One transaction / one shard per attempt: the proposal row never references a missing prompt
            # (meta-prompt and parsed candidate stay restorable) and Phase A writes fewer shard files.
            self.store.write_tables(
                {
                    "prompts": prompts,
                    "proposals": [
                        {
                            "run_id": self.run_id,
                            "seed": seed,
                            "round": t,
                            "attempt": attempt,
                            "meta_prompt_hash": meta_hash,
                            "error_item_idxs": [e.idx for e in shown],
                            "gen_key": rec.gen_key,
                            "parsed_prompt_hash": None if parsed is None else sha256_text(parsed),
                            "valid": int(not violations),
                            "violations": list(violations),
                        }
                    ],
                },
                shard_writer=self.sw,
                stage=STAGE,
            )
            if not violations:
                candidate = parsed
                break
        origin = "proposer"
        if candidate is None:
            candidate, origin = fallback_candidate(st.prompts[inc], seed, t, prior), "fallback"
        self.store.put_slot(
            self.run_id, seed, t, candidate, t, inc, origin, shard_writer=self.sw, stage=STAGE
        )
        cand_res = self.dev_result(seed, t, candidate, PURPOSE_CANDIDATE, t)
        adv = should_advance(cfg, cand_res.n_correct, inc_res.n_correct)
        self.store.put_trajectory_round(
            {
                "run_id": self.run_id,
                "seed": seed,
                "round": t,
                "incumbent_slot": inc,
                "candidate_slot": t,
                "inc_dev_acc": inc_res.acc,
                "cand_dev_acc": cand_res.acc,
                "advanced": int(adv),
                "n_attempts": n_attempts,
                "is_fallback": int(origin == "fallback"),
            },
            shard_writer=self.sw,
            stage=STAGE,
        )
        st.push(candidate, origin, inc, adv, inc_res.acc, cand_res.acc, origin == "fallback")
        self.log(
            f"[trajectory] seed {seed} round {t}: candidate dev {cand_res.acc:.3f} vs incumbent slot {inc} "
            f"{inc_res.acc:.3f} ({origin}, {n_attempts} attempt(s)) -> {'advanced' if adv else 'kept'}"
        )

    def acc0(self, seed: int, round_rows: Mapping[int, dict]) -> float:
        if (seed, 0) in self.cache:
            return self.cache[(seed, 0)].acc
        return _float(round_rows[1]["inc_dev_acc"]) if 1 in round_rows else math.nan


def _rows(store: Store, run_id: str, seed: int) -> tuple[dict[int, dict], dict[int, dict]]:
    slots = {int(r["slot"]): r for r in store.get_slots(run_id, seed)}
    rounds = {int(r["round"]): r for r in store.get_trajectory_rounds(run_id, seed)}
    return slots, rounds


def run_trajectory(
    cfg: ExperimentConfig,
    engine: GenerationEngine,
    store: Store,
    run_id: str,
    dev: DevSplit,
    seeds: Sequence[int] | None = None,
    *,
    log: Callable[[str], None] = print,
    checkpoint: Callable[[], None] | None = None,
) -> dict[int, Trajectory]:
    """Build (or resume) the trajectory of every seed on the DEV split; returns ``{seed: Trajectory}``.

    ``seeds`` defaults to ``cfg.run.seeds``. ``checkpoint()`` is called after every round that committed work.
    Raises ``TypeError`` unless ``dev`` is a :class:`~driftlab.data.DevSplit` (the eval split never reaches
    Phase A).
    """
    if not isinstance(dev, DevSplit):
        raise TypeError(
            f"Phase A runs on the DEV split only: expected a DevSplit, got {type(dev).__name__} "
            "(the eval split must never reach the trajectory or the proposer)"
        )
    seeds = list(cfg.run.seeds if seeds is None else seeds)
    R = cfg.run.rounds
    initial = load_prompt_file(cfg.trajectory.initial_prompt_file)
    pa = _PhaseA(cfg, engine, store, run_id, dev, log)
    states: dict[int, _SeedState] = {}
    rows: dict[int, tuple[dict[int, dict], dict[int, dict]]] = {}
    worked = False
    for seed in seeds:  # round 0
        slot_rows, round_rows = rows[seed] = _rows(store, run_id, seed)
        worked |= 0 not in slot_rows
        pa.check_slot0(seed, slot_rows, initial)
        states[seed] = _SeedState(seed, [initial])
        if 1 not in round_rows:  # slot 0 is the incumbent of round 1 (eager: one batch per round)
            n = engine.n_requested
            pa.dev_result(seed, 0, initial, PURPOSE_DEV, 0)
            worked |= engine.n_requested > n  # cells were committed (even if all cache hits)
    if worked and checkpoint is not None:
        checkpoint()
    for t in range(1, R + 1):
        worked = False
        for seed in seeds:
            slot_rows, round_rows = rows[seed]
            if t in round_rows:
                pa.restore_round(states[seed], t, round_rows[t], slot_rows)
            else:
                pa.compute_round(states[seed], t)
                worked = True
        if worked and checkpoint is not None:
            checkpoint()
    return {seed: states[seed].to_trajectory(pa.acc0(seed, rows[seed][1])) for seed in seeds}


def trajectories_from_store(
    cfg: ExperimentConfig, store: Store, run_id: str, seeds: Sequence[int] | None = None
) -> dict[int, Trajectory]:
    """Rebuild completed trajectories from the store (no model calls, no writes).

    Raises :class:`TrajectoryIncomplete` if some seed lacks slot 0 or a round in 1..R.
    """
    seeds = list(cfg.run.seeds if seeds is None else seeds)
    pa = _PhaseA(cfg, None, store, run_id, None, lambda _msg: None)
    out: dict[int, Trajectory] = {}
    for seed in seeds:
        slot_rows, round_rows = _rows(store, run_id, seed)
        st = _SeedState(seed, [pa.check_slot0(seed, slot_rows, None)])
        for t in range(1, cfg.run.rounds + 1):
            if t not in round_rows:
                raise TrajectoryIncomplete(f"run {run_id!r}: seed {seed} round {t} is not complete")
            pa.restore_round(st, t, round_rows[t], slot_rows)
        out[seed] = st.to_trajectory(pa.acc0(seed, round_rows))
    return out


def trajectory_progress(store: Store, run_id: str, seeds: Sequence[int]) -> dict[int, int]:
    """Completed rounds per seed (``trajectory_rounds`` rows)."""
    done = {int(s): 0 for s in seeds}
    for r in store.query(
        "SELECT seed, COUNT(*) AS n FROM trajectory_rounds WHERE run_id = ? GROUP BY seed", (run_id,)
    ):
        if int(r["seed"]) in done:
            done[int(r["seed"])] = int(r["n"])
    return done


__all__ = [
    "PURPOSE_CANDIDATE",
    "PURPOSE_DEV",
    "PURPOSE_PROPOSER",
    "STAGE",
    "DevResult",
    "TrajectoryIncomplete",
    "dev_tasks",
    "run_trajectory",
    "should_advance",
    "trajectories_from_store",
    "trajectory_progress",
]
