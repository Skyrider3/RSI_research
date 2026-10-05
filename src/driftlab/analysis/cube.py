"""In-memory correctness cube: the single data structure every analysis consumes.

Axes
----
``correct[s, d, k, r, x, n]``  (int8; 1 = correct, 0 = incorrect, -1 = missing)

* ``s`` seed index          -> ``cube.seeds``
* ``d`` decoding index      -> ``cube.decodings``      (e.g. ["greedy", "t02"])
* ``k`` slot (prompt p_k)   -> 0..K-1, K = R + 1
* ``r`` draw index          -> ``cube.draws``          list of (draw_kind, draw), e.g.
                                 ("round", 0) ... ("round", R), ("gt", 0) ... ("gt", G-1), ("audit", a)
* ``x`` extractor index     -> ``cube.extractors``     (e.g. ["v1", "v2"])
* ``n`` eval item index     -> 0..N-1

A cell (s, d, k, ("round", r)) is the logical generation event "prompt p_k answered every eval item under
decoding d at optimization round r". It exists for r >= created_round(k). Cells that share a physical
generation (cache hits, e.g. a non-physical greedy rerun) share ``gen_row`` entries, which is how analyses
detect values that are *identical by construction* (flagged with a double dagger, never reported as a
measured zero). See docs/METHODS.md.

Ground-truth accuracy of slot k under decoding d *at round r* is :meth:`Cube.gt_vec`:
greedy -> the round-r cell itself; sampling -> the mean over independent ("gt", g) draws (unless the
GT mode says otherwise).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

DRAW_ROUND = "round"
DRAW_GT = "gt"
DRAW_AUDIT = "audit"

Draw = tuple[str, int]


class MissingCell(KeyError):
    """Raised when an analysis requests a cell the measurement matrix did not generate."""


@dataclass
class Cube:
    run_id: str
    seeds: list[int]
    decodings: list[str]
    extractors: list[str]
    n_slots: int
    draws: list[Draw]
    n_items: int
    correct: np.ndarray  # int8 (S, D, K, R, X, N), -1 = missing
    gen_row: np.ndarray  # int32 (S, D, K, R, N), -1 = missing; index into gen_keys
    physical: np.ndarray  # bool (S, D, K, R)
    truncated: np.ndarray  # bool (S, D, K, R, N): finish_reason == "length"
    gen_keys: list[str] = field(default_factory=list)
    greedy_decodings: frozenset[str] = frozenset({"greedy"})
    synthetic: bool = False

    # ---------------------------------------------------------------- index helpers
    def __post_init__(self) -> None:
        self._s = {s: i for i, s in enumerate(self.seeds)}
        self._d = {d: i for i, d in enumerate(self.decodings)}
        self._r = {r: i for i, r in enumerate(self.draws)}
        self._x = {x: i for i, x in enumerate(self.extractors)}
        exp = (
            len(self.seeds),
            len(self.decodings),
            self.n_slots,
            len(self.draws),
            len(self.extractors),
            self.n_items,
        )
        if self.correct.shape != exp:
            raise ValueError(f"correct has shape {self.correct.shape}, expected {exp}")

    @property
    def R(self) -> int:
        """Number of optimization rounds (slots are 0..R)."""
        return self.n_slots - 1

    def gt_draws(self) -> list[Draw]:
        return [d for d in self.draws if d[0] == DRAW_GT]

    def is_greedy(self, decoding: str) -> bool:
        return decoding in self.greedy_decodings

    def _idx(self, seed: int, decoding: str, slot: int, draw: Draw) -> tuple[int, int, int, int]:
        try:
            return self._s[seed], self._d[decoding], slot, self._r[tuple(draw)]
        except KeyError as e:  # pragma: no cover - defensive
            raise MissingCell((seed, decoding, slot, draw)) from e

    # ---------------------------------------------------------------- accessors
    def has(self, seed: int, decoding: str, slot: int, draw: Draw) -> bool:
        try:
            s, d, k, r = self._idx(seed, decoding, slot, draw)
        except MissingCell:
            return False
        return bool((self.correct[s, d, k, r, 0] >= 0).all())

    def vec(self, seed: int, decoding: str, slot: int, draw: Draw, extractor: str) -> np.ndarray:
        """Boolean correctness vector (N,) for one cell under one extractor."""
        s, d, k, r = self._idx(seed, decoding, slot, draw)
        v = self.correct[s, d, k, r, self._x[extractor]]
        if (v < 0).any():
            raise MissingCell((seed, decoding, slot, draw, extractor))
        return v.astype(bool)

    def gen_rows(self, seed: int, decoding: str, slot: int, draw: Draw) -> np.ndarray:
        s, d, k, r = self._idx(seed, decoding, slot, draw)
        return self.gen_row[s, d, k, r]

    def same_gen_frac(self, a: tuple[int, str, int, Draw], b: tuple[int, str, int, Draw]) -> float:
        """Fraction of items on which two cells point at the same physical generation."""
        ga, gb = self.gen_rows(*a), self.gen_rows(*b)
        ok = (ga >= 0) & (gb >= 0)
        if not ok.any():
            return float("nan")
        return float((ga[ok] == gb[ok]).mean())

    def is_physical(self, seed: int, decoding: str, slot: int, draw: Draw) -> bool:
        s, d, k, r = self._idx(seed, decoding, slot, draw)
        return bool(self.physical[s, d, k, r])

    def gt_vec(
        self,
        seed: int,
        decoding: str,
        slot: int,
        round_: int,
        extractor: str,
        mode: str = "independent_draw",
    ) -> np.ndarray:
        """Per-item ground-truth correctness *rate* (float, N) of slot under decoding at round_.

        * greedy decodings: the ("round", round_) cell (deterministic up to engine noise);
        * sampling + ``independent_draw``: mean over the ("gt", g) draws (independent of decision draws);
        * sampling + ``same_draw``: the ("round", round_) cell (couples decision and GT; sensitivity only).
        ``split_half`` is handled by the caller (it restricts items, not draws) and uses independent draws.
        """
        if self.is_greedy(decoding) or mode == "same_draw":
            return self.vec(seed, decoding, slot, (DRAW_ROUND, round_), extractor).astype(float)
        gts = self.gt_draws()
        if not gts:
            raise MissingCell((seed, decoding, slot, "gt", extractor))
        return np.mean([self.vec(seed, decoding, slot, g, extractor) for g in gts], axis=0)

    def gt_acc(
        self,
        seed: int,
        decoding: str,
        slot: int,
        round_: int,
        extractor: str,
        mode: str = "independent_draw",
        items: np.ndarray | None = None,
    ) -> float:
        v = self.gt_vec(seed, decoding, slot, round_, extractor, mode)
        if items is not None:
            v = v[items]
        return float(v.mean())

    def trunc_rate(self, seed: int, decoding: str, slot: int, draw: Draw) -> float:
        s, d, k, r = self._idx(seed, decoding, slot, draw)
        return float(self.truncated[s, d, k, r].mean())


@dataclass
class Trajectory:
    """The shared prompt chain p_0..p_R of one seed (built on the DEV split only).

    * ``inc_slot[t]`` (t = 0..R): trajectory incumbent when candidate t is proposed/evaluated;
      ``inc_slot[0] = 0`` by convention, ``inc_slot[1] = 0``; ``inc_slot[t+1] = t`` if candidate t advanced,
      else ``inc_slot[t]``.
    * the reference *created at round i* is the output of ``inc_slot[i]`` at draw ("round", i).
    """

    seed: int
    prompts: list[str]  # text of slot k
    prompt_hashes: list[str]
    created_round: list[int]  # slot k -> round it was created (k)
    inc_slot: list[int]  # length R+1
    advanced: list[bool]  # length R+1; advanced[0] = False
    inc_dev_acc: list[float]  # length R+1 (NaN at 0)
    cand_dev_acc: list[float]  # length R+1 (dev accuracy of slot t; slot 0 at index 0)
    is_fallback: list[bool]  # length R+1
    origin: list[str] = field(default_factory=list)  # slot k -> initial | proposer | fallback

    @property
    def R(self) -> int:
        return len(self.prompts) - 1

    def incumbent_after(self, t: int) -> int:
        """Trajectory incumbent after round t has been decided."""
        return t if (t >= 1 and self.advanced[t]) else self.inc_slot[t]


def make_trajectory(
    seed: int, prompts: Sequence[str], advanced: Sequence[bool], dev_acc: Sequence[float] | None = None
) -> Trajectory:
    """Build a Trajectory from prompt texts and per-round advance flags (advanced[0] ignored)."""
    from driftlab.keys import sha256_text

    R = len(prompts) - 1
    adv = [False] + [bool(a) for a in list(advanced)[1:]]
    inc = [0] * (R + 1)
    for t in range(1, R + 1):
        inc[t] = 0 if t == 1 else ((t - 1) if adv[t - 1] else inc[t - 1])
    dev = list(dev_acc) if dev_acc is not None else [float("nan")] * (R + 1)
    return Trajectory(
        seed=seed,
        prompts=list(prompts),
        prompt_hashes=[sha256_text(p) for p in prompts],
        created_round=list(range(R + 1)),
        inc_slot=inc,
        advanced=adv,
        inc_dev_acc=[float("nan")] + [dev[inc[t]] for t in range(1, R + 1)],
        cand_dev_acc=dev,
        is_fallback=[False] * (R + 1),
        origin=["initial"] + ["proposer"] * R,
    )


def empty_cube(
    run_id: str,
    seeds: Sequence[int],
    decodings: Sequence[str],
    extractors: Sequence[str],
    n_slots: int,
    draws: Sequence[Draw],
    n_items: int,
    greedy_decodings: Sequence[str] = ("greedy",),
) -> Cube:
    """All-missing cube; used by the loader and by tests that hand-build matrices."""
    S, D, K, R, X, N = len(seeds), len(decodings), n_slots, len(draws), len(extractors), n_items
    return Cube(
        run_id=run_id,
        seeds=list(seeds),
        decodings=list(decodings),
        extractors=list(extractors),
        n_slots=n_slots,
        draws=[tuple(d) for d in draws],
        n_items=n_items,
        correct=np.full((S, D, K, R, X, N), -1, dtype=np.int8),
        gen_row=np.full((S, D, K, R, N), -1, dtype=np.int32),
        physical=np.zeros((S, D, K, R), dtype=bool),
        truncated=np.zeros((S, D, K, R, N), dtype=bool),
        greedy_decodings=frozenset(greedy_decodings),
    )


def set_cell(
    cube: Cube,
    seed: int,
    decoding: str,
    slot: int,
    draw: Draw,
    correct_by_extractor: dict[str, Sequence[int] | np.ndarray],
    gen_rows: Sequence[int] | np.ndarray | None = None,
    physical: bool = False,
    truncated: Sequence[bool] | np.ndarray | None = None,
) -> None:
    """Fill one cell (test/loader helper). Missing gen_rows get unique fresh ids."""
    s, d, k, r = cube._idx(seed, decoding, slot, draw)
    for x, vals in correct_by_extractor.items():
        cube.correct[s, d, k, r, cube._x[x]] = np.asarray(vals, dtype=np.int8)
    if gen_rows is None:
        start = len(cube.gen_keys)
        cube.gen_keys.extend(f"auto:{start + i}" for i in range(cube.n_items))
        gen_rows = np.arange(start, start + cube.n_items)
    cube.gen_row[s, d, k, r] = np.asarray(gen_rows, dtype=np.int32)
    cube.physical[s, d, k, r] = physical
    if truncated is not None:
        cube.truncated[s, d, k, r] = np.asarray(truncated, dtype=bool)


def load_cube(
    db_path: str, run_id: str | None = None, split: str = "test"
):  # pragma: no cover - implemented in loader
    from driftlab.analysis.loader import load_cube as _impl

    return _impl(db_path, run_id=run_id, split=split)


def load_trajectories(db_path: str, run_id: str | None = None):  # pragma: no cover - implemented in loader
    from driftlab.analysis.loader import load_trajectories as _impl

    return _impl(db_path, run_id=run_id)
