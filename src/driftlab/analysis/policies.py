"""Dry-run reference-refresh policy simulation (docs/ARCHITECTURE.md section 6.3): source of T3, T5, T6, T9.

Every policy keeps a *shadow incumbent* and a *reference* (stored outputs + the scores they had at storage
time) and walks the shared trajectory's candidates ``t = 1..R`` under a predefined environment schedule that
is independent of all decisions. Nothing is ever promoted for real; all values come from the cube, so the
simulation makes no model calls. Model-call *costs* are logical counts under a :class:`CostConvention`.

Policies (:func:`parse_policy`)
-------------------------------
* ``P1`` frozen: the reference is the round-0 output of slot 0 forever (never refresh, never adopt). It mixes
  environment staleness with prompt staleness (it also accepts candidates that beat the starting prompt but
  not the current incumbent), matching the teammate's footnote.
* ``P1b`` frozen + adopt-on-promote: on acceptance the candidate's already-computed outputs become the
  reference at zero cost, so ``FAR(P1) - FAR(P1b)`` isolates prompt staleness.
* ``P2`` per-batch refresh (every round), ``P3`` environment-triggered (env fingerprint differs from the
  reference's), ``P4_k<k>`` age-triggered (``t - created_round >= k``), ``P5`` component-aware (regenerate if
  decoding/model/engine/max_new_tokens changed; re-score the stored text at zero model calls if only the
  extractor changed), ``ORACLE`` (no reference: accept iff GT(candidate) > GT(incumbent); costs N per round),
  ``FIXEDAGE_k<k>`` (ablation A4: the reference is the output of the shadow incumbent *as it was when
  candidate ``r0 = max(t - k, 0)`` was evaluated*, generated at round ``r0`` under that round's environment;
  ``FIXEDAGE_k0`` therefore equals ``P2``). All policies except ``P1``, ``ORACLE`` and ``FIXEDAGE`` adopt on
  promotion (``FIXEDAGE`` replaces its reference every round, so an adopted reference would never be used).

Ground truth
------------
``false_accept = accepted and GT(cand) <= GT(incumbent)`` with both GTs under the round's environment
(``cube.gt_acc``; greedy -> the round-t cell, sampling -> independent ``("gt", g)`` draws by default).
Under greedy decoding with a fresh reference (P2) and any of the three promotion rules (``net_win`` with
margin >= 0, ``win_rate``, ``mcnemar``), acceptance requires ``wins > losses`` on the same items the GT is
computed on, i.e. ``sum(cand) > sum(ref) = N * GT(inc)``, so P2's FAR is **0 by construction**; this is a
property of the definitions, not a finding. The same holds for *any* policy in a round whose reference is
the incumbent's cached (non-physical) greedy generation under the current extractor (e.g. P1b/P3/P5 within
an unchanged greedy segment); every such round is flagged ``RoundDecision.gt_coupled`` and counted in
``summarize_policies`` (``n_gt_coupled``, ``n_accepted_gt_coupled``) so reports can footnote it. Under
sampling the decision and GT use independent draws, and ``split_half`` decides on items ``[0, N/2)`` and
measures GT on ``[N/2, N)`` (explicit ``items`` / ``gt_items`` must then both be given and be disjoint).

Costs (``teammate_v1``) per round: ``n_dev`` (incumbent dev run) + 1 (proposer) + N (candidate eval), plus N
per reference regeneration; with R = 11, N = n_dev = 200 and the schedule ``{0: E1, 4: E2, 8: E4}`` this gives
P1 4411, P2 6611, P3 4811, P5 4611 (pinned by tests). ``full`` also counts candidate dev runs, every proposer
attempt and the initial reference.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from driftlab.analysis.cube import DRAW_ROUND, Cube, Draw, MissingCell, Trajectory
from driftlab.analysis.metrics import decide, mean_sd, paired, wilson
from driftlab.environments import DEFAULT_DECODINGS, Environment, EnvSchedule, build_environments, diff

PolicyKind = Literal[
    "frozen",
    "frozen_adopt",
    "per_batch",
    "env_triggered",
    "age_triggered",
    "component_aware",
    "oracle",
    "fixed_age",
]
RefSource = Literal["initial", "refresh", "adopt", "rescore", "fixed_age"]
RefreshKind = Literal["none", "refresh", "rescore", "fixed_age", "initial"]

# Logical model-call purposes, in reporting order.
CALL_PURPOSES: tuple[str, ...] = (
    "trajectory_dev",
    "proposer",
    "candidate_dev",
    "candidate_eval",
    "reference_init",
    "reference_refresh",
    "rescore",
    "oracle_gt",
)
# Component changes that force P5 to regenerate (an extractor-only change is re-scored for free).
REGENERATE_COMPONENTS: frozenset[str] = frozenset({"decoding", "model", "engine", "max_new_tokens"})
GT_EPS = 1e-9  # tolerance for "gt_cand <= gt_inc" (same as the all-pairs analysis)
NO_REF = -1  # ref_slot / ref_round / ref_age of ORACLE rows (ORACLE keeps no reference)


# --------------------------------------------------------------------------- policy specs


@dataclass(frozen=True)
class PolicySpec:
    """A reference-refresh policy. Build with :func:`parse_policy`."""

    name: str
    kind: PolicyKind
    adopt_on_promote: bool
    k: int | None = None
    label: str = ""
    refresh_rule: str = ""

    def __post_init__(self) -> None:
        if self.kind in ("age_triggered", "fixed_age") and (self.k is None or int(self.k) < 0):
            raise ValueError(f"policy {self.name!r} of kind {self.kind!r} needs an integer k >= 0")


_FIXED: dict[str, PolicySpec] = {
    "P1": PolicySpec("P1", "frozen", False, None, "Frozen reference", "Never refresh"),
    "P1b": PolicySpec(
        "P1b",
        "frozen_adopt",
        True,
        None,
        "Frozen, adopt on promote",
        "Never refresh; adopt the candidate's outputs on promotion",
    ),
    "P2": PolicySpec("P2", "per_batch", True, None, "Per-batch refresh", "Refresh every batch"),
    "P3": PolicySpec(
        "P3",
        "env_triggered",
        True,
        None,
        "Environment-triggered refresh",
        "Refresh after detected environment changes",
    ),
    "P5": PolicySpec(
        "P5",
        "component_aware",
        True,
        None,
        "Component-aware refresh (re-score on extractor change, regenerate otherwise)",
        "Regenerate on a decoding/model/engine/token-cap change; re-score the stored text (0 calls) "
        "on an extractor-only change",
    ),
    "ORACLE": PolicySpec(
        "ORACLE",
        "oracle",
        False,
        None,
        "Ground-truth control",
        "No reference; accept iff ground-truth accuracy improves",
    ),
}
_P4_RE = re.compile(r"^P4_k(\d+)$", re.I)
_FIXEDAGE_RE = re.compile(r"^FIXEDAGE_k(\d+)$", re.I)


def parse_policy(name: str | PolicySpec) -> PolicySpec:
    """``"P1" | "P1b" | "P2" | "P3" | "P4_k<k>" | "P5" | "ORACLE" | "FIXEDAGE_k<k>"`` -> :class:`PolicySpec`.

    Names are matched case-insensitively and returned in canonical spelling; a PolicySpec passes through.
    """
    if isinstance(name, PolicySpec):
        return name
    raw = str(name).strip()
    for key, spec in _FIXED.items():
        if raw.lower() == key.lower():
            return spec
    m = _P4_RE.match(raw)
    if m:
        k = int(m.group(1))
        return PolicySpec(
            f"P4_k{k}",
            "age_triggered",
            True,
            k,
            f"Age-triggered refresh (every {k} rounds)",
            f"Refresh when the reference is >= {k} rounds old",
        )
    m = _FIXEDAGE_RE.match(raw)
    if m:
        k = int(m.group(1))
        return PolicySpec(
            f"FIXEDAGE_k{k}",
            "fixed_age",
            False,
            k,
            f"Fixed reference age ({k} round{'' if k == 1 else 's'})",
            f"Reference = incumbent output stored {k} round{'' if k == 1 else 's'} earlier",
        )
    raise ValueError(
        f"unknown policy {name!r}; expected P1, P1b, P2, P3, P4_k<k>, P5, ORACLE or FIXEDAGE_k<k>"
    )


# --------------------------------------------------------------------------- cost conventions


@dataclass(frozen=True)
class CostConvention:
    """How logical model calls are counted (docs/ARCHITECTURE.md section 6.3).

    ``rescore_cost`` is charged per re-score event (calls purpose ``rescore``); re-scoring needs no model
    call, so it is 0 in both built-in conventions.
    """

    name: str
    count_inc_dev: bool = True
    count_candidate_dev: bool = False
    proposer: Literal["one_per_round", "attempts"] = "one_per_round"
    count_candidate_eval: bool = True
    count_reference_init: bool = False
    rescore_cost: int = 0


CONVENTIONS: dict[str, CostConvention] = {
    "teammate_v1": CostConvention("teammate_v1"),
    "full": CostConvention("full", count_candidate_dev=True, proposer="attempts", count_reference_init=True),
}


def get_convention(conv: str | CostConvention | None) -> CostConvention:
    """Resolve a convention name (``None`` -> ``teammate_v1``)."""
    if conv is None:
        return CONVENTIONS["teammate_v1"]
    if isinstance(conv, CostConvention):
        return conv
    try:
        return CONVENTIONS[str(conv)]
    except KeyError as e:
        raise ValueError(f"unknown cost convention {conv!r}; known: {sorted(CONVENTIONS)}") from e


# --------------------------------------------------------------------------- records


@dataclass
class RefSnapshot:
    """One reference version kept by a policy (the Reference Manager's lifecycle rows).

    ``created_round`` is the round the stored outputs were generated (a re-score keeps it, so age keeps
    counting); ``env_id`` / ``extractor_at_storage`` describe the scores currently held (a re-score updates
    both). ``acc_at_storage`` is the mean of the held scores over all N items.
    """

    ref_id: str
    slot: int
    created_round: int
    env_id: str
    decoding: str
    extractor_at_storage: str
    source: RefSource
    retired_round: int | None
    acc_at_storage: float
    rescored_round: int | None = None
    parent_id: str | None = None


@dataclass
class _LiveRef:
    """The in-simulation reference: snapshot metadata + the scores vector and the draw it came from."""

    snap: RefSnapshot
    env: Environment
    scores: np.ndarray  # bool (N,)
    draw: Draw


@dataclass
class RoundDecision:
    """One policy decision on candidate ``round`` (= ``cand_slot``) of one seed.

    ``calls`` holds this round's logical calls by purpose (every key of :data:`CALL_PURPOSES`).
    ORACLE rows keep no reference: ``ref_id = ref_env = ""``, ``ref_slot = ref_round = ref_age = NO_REF`` and
    ``wins = losses = ties = 0``. ``ref_env_stale`` is True when the comparator's environment fingerprint
    differs from the round's environment at decision time. ``refresh_kind`` is what happened to the
    reference *this round* before the comparison: ``none | refresh | rescore | fixed_age`` (``initial`` is
    reserved for round 0, which has no decision row).

    ``gt_coupled`` is True when the decision *is* the ground-truth comparison by construction: the reference
    scores are the very generations, extractor and items that define ``gt_inc`` (and the candidate's decision
    cell defines ``gt_cand``: greedy or ``same_draw`` GT), so ``wins - losses = n * (gt_cand - gt_inc)`` and,
    for any rule needing ``wins > losses``, a false accept is impossible. ORACLE rows are always coupled. Such
    zeros must be footnoted as identical by construction (like the all-pairs ``gt_coupled`` flag), never
    reported as measured.
    """

    seed: int
    policy: str
    round: int
    env_id: str
    ref_id: str
    ref_slot: int
    ref_round: int
    ref_env: str
    ref_age: int
    refresh_kind: RefreshKind
    wins: int
    losses: int
    ties: int
    n: int
    accepted: bool
    gt_cand: float
    gt_inc: float
    false_accept: bool
    inc_before: int
    inc_after: int
    cand_slot: int
    calls: dict[str, int]
    ref_source: str = ""
    ref_decoding: str = ""
    ref_extractor: str = ""
    ref_env_stale: bool = False
    gt_coupled: bool = False


@dataclass
class PolicyRun:
    """Result of :func:`simulate` for one (seed, policy)."""

    seed: int
    policy: PolicySpec
    rounds: list[RoundDecision]
    refs: list[RefSnapshot]
    calls: dict[str, int]
    refreshes: int
    rescores: int
    final_inc: int
    final_acc_canonical: float
    final_acc_final_env: float
    convention: str = "teammate_v1"
    gt_mode: str = "independent_draw"
    canonical_env: str = "E1"
    schedule: dict[int, str] = field(default_factory=dict)

    @property
    def total_calls(self) -> int:
        return int(sum(self.calls.values()))

    @property
    def candidate_generation_calls(self) -> int:
        c = self.calls
        return int(c.get("trajectory_dev", 0) + c.get("proposer", 0) + c.get("candidate_dev", 0))

    @property
    def evaluation_calls(self) -> int:
        return int(self.calls.get("candidate_eval", 0) + self.calls.get("oracle_gt", 0))

    @property
    def reference_calls(self) -> int:
        return int(self.calls.get("reference_init", 0) + self.calls.get("reference_refresh", 0))

    @property
    def n_evaluated(self) -> int:
        return len(self.rounds)

    @property
    def n_accepted(self) -> int:
        return int(sum(r.accepted for r in self.rounds))

    @property
    def n_false_accepts(self) -> int:
        return int(sum(r.false_accept for r in self.rounds))

    @property
    def far(self) -> float:
        """False-accept rate of this seed: false accepts / accepts (NaN with no accepts)."""
        n = self.n_accepted
        return self.n_false_accepts / n if n else float("nan")

    @property
    def n_gt_coupled(self) -> int:
        """Rounds whose decision equals the GT comparison by construction (see ``RoundDecision``)."""
        return int(sum(r.gt_coupled for r in self.rounds))

    @property
    def n_accepted_gt_coupled(self) -> int:
        """Accepts that could not have been false accepts by construction."""
        return int(sum(r.accepted and r.gt_coupled for r in self.rounds))


# --------------------------------------------------------------------------- helpers


def as_environments(envs: Mapping[str, object] | None) -> dict[str, Environment]:
    """``env_id -> Environment``.

    Accepts Environment objects, ``(decoding_id, extractor[, description])`` tuples or objects with
    ``decoding`` (a ``Decoding`` or an id) and ``extractor`` attributes; ids resolve through
    :data:`driftlab.environments.DEFAULT_DECODINGS`. ``None`` -> :func:`build_environments`.
    """
    if envs is None:
        return build_environments()
    out: dict[str, Environment] = {}
    for eid, e in envs.items():
        if isinstance(e, Environment):
            out[str(eid)] = e
            continue
        if isinstance(e, (tuple, list)):
            if len(e) < 2:
                raise ValueError(f"environment {eid!r} must be (decoding_id, extractor), got {e!r}")
            dec, ext, desc = e[0], e[1], (str(e[2]) if len(e) > 2 else "")
        elif hasattr(e, "decoding") and hasattr(e, "extractor"):
            dec, ext, desc = e.decoding, e.extractor, str(getattr(e, "description", ""))
        else:
            raise TypeError(f"cannot read decoding/extractor of environment {eid!r}: {e!r}")
        if not hasattr(dec, "params"):
            if str(dec) not in DEFAULT_DECODINGS:
                raise ValueError(f"unknown decoding {dec!r} for environment {eid!r}; pass an Environment")
            dec = DEFAULT_DECODINGS[str(dec)]
        out[str(eid)] = Environment(id=str(eid), decoding=dec, extractor=str(ext), description=desc)
    return out


def _as_schedule(schedule: EnvSchedule | Mapping[int, str] | None) -> EnvSchedule:
    if schedule is None:
        return EnvSchedule()
    if isinstance(schedule, EnvSchedule):
        return schedule
    return EnvSchedule({int(k): str(v) for k, v in dict(schedule).items()})


def _zero_calls() -> dict[str, int]:
    return dict.fromkeys(CALL_PURPOSES, 0)


def _resolve_items(
    n_items: int, gt_mode: str, items: np.ndarray | None, gt_items: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray | None, str]:
    """(decision items, GT items or None for all, mode passed to ``cube.gt_acc``).

    ``split_half`` without explicit items decides on ``[0, N/2)`` and measures GT on ``[N/2, N)`` with
    independent draws. Explicit (possibly resampled, with repeats) index vectors are used as given.
    """
    mode = "independent_draw" if gt_mode == "split_half" else gt_mode
    if gt_mode == "split_half":
        if items is None and gt_items is None:
            h = n_items // 2
            return np.arange(h), np.arange(h, n_items), mode
        # One half alone would silently decide (or measure GT) on ALL items, re-coupling decision and GT.
        if items is None or gt_items is None:
            raise ValueError("split_half needs both items and gt_items (disjoint item sets), or neither")
        dec, gt = np.asarray(items, dtype=np.int64), np.asarray(gt_items, dtype=np.int64)
        if np.intersect1d(dec, gt).size:
            raise ValueError("split_half decision items and GT items must be disjoint")
        return dec, gt, mode
    dec = np.arange(n_items) if items is None else np.asarray(items, dtype=np.int64)
    gt = None if gt_items is None else np.asarray(gt_items, dtype=np.int64)
    return dec, gt, mode


def _same_generations(
    cube: Cube, seed: int, a: tuple[str, int, Draw], b: tuple[str, int, Draw], items: np.ndarray
) -> bool:
    """True when cells ``a`` and ``b`` (``(decoding, slot, draw)``) are the same physical generations.

    Mirrors the all-pairs rule: the identical cell always is; otherwise every compared item must point at the
    same ``gen_row``; without gen rows, fall back to the matrix semantics of ARCHITECTURE section 4 (a
    non-physical cell of the same decoding is a cache hit of the slot's creation generation).
    """
    if a[0] == b[0] and a[1] == b[1] and tuple(a[2]) == tuple(b[2]):
        return True
    if a[0] != b[0] or a[1] != b[1]:
        return False
    ga = cube.gen_rows(seed, *a)[items]
    gb = cube.gen_rows(seed, *b)[items]
    ok = (ga >= 0) & (gb >= 0)
    if ok.any():
        return bool((ga[ok] == gb[ok]).all())
    if cube.is_physical(seed, *b):
        return False
    return tuple(a[2]) == (DRAW_ROUND, int(a[1])) or not cube.is_physical(seed, *a)


def item_sets(n_items: int, gt_mode: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """``(items, gt_items)`` to pass to :func:`simulate` for a plan's GT mode (``None`` = all items)."""
    if gt_mode == "split_half":
        h = n_items // 2
        return np.arange(h), np.arange(h, n_items)
    return None, None


def _env(envmap: Mapping[str, Environment], env_id: str) -> Environment:
    try:
        return envmap[env_id]
    except KeyError as e:
        raise ValueError(f"schedule uses environment {env_id!r}, not among {sorted(envmap)}") from e


# --------------------------------------------------------------------------- simulation


def simulate(
    cube: Cube,
    traj: Trajectory,
    schedule: EnvSchedule,
    envs: Mapping[str, Environment],
    policy: PolicySpec | str,
    rule,
    gt_mode: str = "independent_draw",
    canonical_env: str = "E1",
    conv: CostConvention | str = CONVENTIONS["teammate_v1"],
    n_dev: int = 200,
    extractor_tags: Mapping[str, str] | None = None,
    proposer_attempts: Mapping[int, int] | None = None,
    items: np.ndarray | None = None,
    gt_items: np.ndarray | None = None,
) -> PolicyRun:
    """Simulate one policy on one seed's trajectory (dry run; see the module docstring).

    ``rule`` is a :class:`driftlab.config.PromotionRule`. ``items`` restricts (or resamples, repeats allowed)
    the decision items, ``gt_items`` the ground-truth items; ``gt_mode`` is passed to ``cube.gt_acc``
    (``split_half`` is resolved to item halves when no items are given; explicit items must then come as both
    ``items`` and ``gt_items``, disjoint, else ``ValueError``). ``proposer_attempts`` maps round ->
    proposer attempts (used by conventions with ``proposer="attempts"``; missing rounds count 1).
    Raises :class:`MissingCell` when a needed cell is absent.
    """
    spec = parse_policy(policy)
    cost = get_convention(conv)
    envmap = as_environments(envs)
    sched = _as_schedule(schedule)
    seed, R, N = int(traj.seed), int(traj.R), int(cube.n_items)
    if cube.n_slots < R + 1:
        raise ValueError(f"cube has {cube.n_slots} slots but the trajectory needs {R + 1}")
    dec_items, gt_idx, mode = _resolve_items(N, gt_mode, items, gt_items)
    env_at = [_env(envmap, sched.env_at(t)) for t in range(R + 1)]
    canon = _env(envmap, str(canonical_env))
    tags = extractor_tags
    # GT is computed on the same multiset of items as the decision (sums are order-free): precondition of a
    # by-construction coupling.
    same_items = np.array_equal(np.sort(dec_items), np.sort(gt_idx) if gt_idx is not None else np.arange(N))

    def vec(env: Environment, slot: int, round_: int) -> np.ndarray:
        return cube.vec(seed, env.decoding.id, slot, (DRAW_ROUND, round_), env.extractor)

    def gt(env: Environment, slot: int, round_: int) -> float:
        return cube.gt_acc(
            seed, env.decoding.id, slot, round_=round_, extractor=env.extractor, mode=mode, items=gt_idx
        )

    refs: list[RefSnapshot] = []

    def new_ref(
        slot: int,
        round_: int,
        env: Environment,
        scores: np.ndarray,
        source: RefSource,
        draw: Draw,
        decoding: str | None = None,
        rescored_round: int | None = None,
        parent: str | None = None,
    ) -> _LiveRef:
        snap = RefSnapshot(
            ref_id=f"{spec.name}:s{seed}:{len(refs)}",
            slot=int(slot),
            created_round=int(round_),
            env_id=env.id,
            decoding=decoding if decoding is not None else env.decoding.id,
            extractor_at_storage=env.extractor,
            source=source,
            retired_round=None,
            acc_at_storage=float(np.mean(scores)) if len(scores) else float("nan"),
            rescored_round=rescored_round,
            parent_id=parent,
        )
        refs.append(snap)
        return _LiveRef(snap=snap, env=env, scores=np.asarray(scores, dtype=bool), draw=draw)

    calls = _zero_calls()
    refreshes = rescores = 0
    inc = 0
    inc_hist: list[int] = [0]  # inc_hist[t] = shadow incumbent when candidate t is evaluated
    ref: _LiveRef | None = None
    if spec.kind != "oracle":
        ref = new_ref(0, 0, env_at[0], vec(env_at[0], 0, 0), "initial", (DRAW_ROUND, 0))
        if cost.count_reference_init:
            calls["reference_init"] += N

    rounds: list[RoundDecision] = []
    for t in range(1, R + 1):
        env = env_at[t]
        rc = _zero_calls()
        if cost.count_inc_dev:
            rc["trajectory_dev"] += int(n_dev)
        if cost.proposer == "attempts":
            rc["proposer"] += int((proposer_attempts or {}).get(t, 1))
        else:
            rc["proposer"] += 1
        if cost.count_candidate_dev:
            rc["candidate_dev"] += int(n_dev)
        if cost.count_candidate_eval:
            rc["candidate_eval"] += N
        inc_hist.append(inc)
        cand = vec(env, t, t)
        gt_cand, gt_inc = gt(env, t, t), gt(env, inc, t)
        refresh_kind: RefreshKind = "none"

        if ref is None:  # ORACLE
            rc["oracle_gt"] += N
            accepted = bool(gt_cand > gt_inc + GT_EPS)
            wins = losses = ties = 0
            n = int(len(dec_items))
        else:
            action = _refresh_action(spec, ref, env, t, tags)
            if action == "refresh":
                ref.snap.retired_round = t
                ref = new_ref(inc, t, env, vec(env, inc, t), "refresh", (DRAW_ROUND, t))
                rc["reference_refresh"] += N
                refreshes += 1
                refresh_kind = "refresh"
            elif action == "rescore":
                old = ref
                old.snap.retired_round = t
                scores = cube.vec(seed, old.snap.decoding, old.snap.slot, old.draw, env.extractor)
                ref = new_ref(
                    old.snap.slot,
                    old.snap.created_round,
                    env,
                    scores,
                    "rescore",
                    old.draw,
                    decoding=old.snap.decoding,
                    rescored_round=t,
                    parent=old.snap.ref_id,
                )
                rc["rescore"] += int(cost.rescore_cost)
                rescores += 1
                refresh_kind = "rescore"
            elif action == "fixed_age":
                r0 = max(t - int(spec.k or 0), 0)
                slot0, env0 = inc_hist[r0], env_at[r0]
                cur = ref.snap
                if not (cur.slot == slot0 and cur.created_round == r0 and cur.env_id == env0.id):
                    cur.retired_round = t
                    ref = new_ref(slot0, r0, env0, vec(env0, slot0, r0), "fixed_age", (DRAW_ROUND, r0))
                    rc["reference_refresh"] += N
                    refreshes += 1
                    refresh_kind = "fixed_age"
            p = paired(cand[dec_items], ref.scores[dec_items])
            wins, losses, ties, n = p.wins, p.losses, p.ties, p.n
            accepted = bool(decide(wins, losses, n, rule))

        if ref is None:
            coupled = True  # ORACLE decides on the ground truth itself
        else:
            coupled = bool(
                same_items
                and (cube.is_greedy(env.decoding.id) or mode == "same_draw")
                and ref.snap.slot == inc
                and ref.snap.extractor_at_storage == env.extractor
                and _same_generations(
                    cube,
                    seed,
                    (ref.snap.decoding, ref.snap.slot, ref.draw),
                    (env.decoding.id, inc, (DRAW_ROUND, t)),
                    dec_items,
                )
            )
        false_accept = bool(accepted and gt_cand <= gt_inc + GT_EPS)
        inc_after = t if accepted else inc
        rounds.append(
            RoundDecision(
                seed=seed,
                policy=spec.name,
                round=t,
                env_id=env.id,
                ref_id=ref.snap.ref_id if ref else "",
                ref_slot=ref.snap.slot if ref else NO_REF,
                ref_round=ref.snap.created_round if ref else NO_REF,
                ref_env=ref.snap.env_id if ref else "",
                ref_age=(t - ref.snap.created_round) if ref else NO_REF,
                refresh_kind=refresh_kind,
                wins=int(wins),
                losses=int(losses),
                ties=int(ties),
                n=int(n),
                accepted=accepted,
                gt_cand=float(gt_cand),
                gt_inc=float(gt_inc),
                false_accept=false_accept,
                inc_before=int(inc),
                inc_after=int(inc_after),
                cand_slot=t,
                calls=rc,
                ref_source=ref.snap.source if ref else "",
                ref_decoding=ref.snap.decoding if ref else "",
                ref_extractor=ref.snap.extractor_at_storage if ref else "",
                ref_env_stale=bool(ref and ref.env.fingerprint(tags) != env.fingerprint(tags)),
                gt_coupled=coupled,
            )
        )
        for key, v in rc.items():
            calls[key] += v
        if accepted:
            inc = t
            if spec.adopt_on_promote and ref is not None:
                ref.snap.retired_round = t
                ref = new_ref(t, t, env, cand, "adopt", (DRAW_ROUND, t))

    return PolicyRun(
        seed=seed,
        policy=spec,
        rounds=rounds,
        refs=refs,
        calls=calls,
        refreshes=refreshes,
        rescores=rescores,
        final_inc=int(inc),
        final_acc_canonical=float(gt(canon, inc, R)),
        final_acc_final_env=float(gt(env_at[R], inc, R)),
        convention=cost.name,
        gt_mode=str(gt_mode),
        canonical_env=canon.id,
        schedule={int(k): str(v) for k, v in sched.changes.items()},
    )


def _refresh_action(
    spec: PolicySpec, ref: _LiveRef, env: Environment, t: int, tags: Mapping[str, str] | None
) -> str:
    """``"none" | "refresh" | "rescore" | "fixed_age"`` for round ``t`` (before the comparison)."""
    kind = spec.kind
    if kind in ("frozen", "frozen_adopt"):
        return "none"
    if kind == "per_batch":
        return "refresh"
    if kind == "env_triggered":
        return "refresh" if env.fingerprint(tags) != ref.env.fingerprint(tags) else "none"
    if kind == "age_triggered":
        return "refresh" if t - ref.snap.created_round >= int(spec.k or 0) else "none"
    if kind == "component_aware":
        changed = set(diff(ref.env, env, tags))
        if changed & REGENERATE_COMPONENTS:
            return "refresh"
        return "rescore" if "extractor" in changed else "none"
    if kind == "fixed_age":
        return "fixed_age"
    raise ValueError(f"policy kind {kind!r} has no refresh rule")


def as_trajectories(trajs: Mapping[int, Trajectory] | Sequence[Trajectory]) -> dict[int, Trajectory]:
    """``seed -> Trajectory`` from a mapping or a sequence of trajectories."""
    if isinstance(trajs, Mapping):
        return {int(k): v for k, v in trajs.items()}
    return {int(t.seed): t for t in trajs}


def simulate_all(
    cube: Cube,
    trajs: Mapping[int, Trajectory] | Sequence[Trajectory],
    plan,
    envs: Mapping[str, Environment] | None,
    n_dev: int,
    extractor_tags: Mapping[str, str] | None = None,
    proposer_attempts_by_seed: Mapping[int, Mapping[int, int]] | None = None,
    policies: Sequence[str] | None = None,
    schedule: EnvSchedule | None = None,
    *,
    conv: CostConvention | str | None = None,
    on_missing: Literal["raise", "skip"] = "raise",
) -> list[PolicyRun]:
    """Simulate every policy (default ``plan.policies``) on every seed that has a trajectory.

    Uses ``plan.promotion_rule``, ``plan.gt`` (``split_half`` -> item halves), ``plan.cost_convention``
    (override with ``conv``) and ``plan.env_schedule()`` (override with ``schedule``). Runs are ordered
    policy-major, then by ``cube.seeds``. ``on_missing="skip"`` drops (with a warning) a (seed, policy)
    whose cells are missing instead of raising :class:`MissingCell`.
    """
    names = list(policies) if policies is not None else list(plan.policies)
    sched = _as_schedule(schedule if schedule is not None else plan.env_schedule())
    cost = get_convention(conv if conv is not None else plan.cost_convention)
    envmap = as_environments(envs)
    tmap = as_trajectories(trajs)
    items, gt_items = item_sets(cube.n_items, plan.gt.mode)
    attempts = proposer_attempts_by_seed or {}
    out: list[PolicyRun] = []
    for name in names:
        spec = parse_policy(name)
        for seed in cube.seeds:
            traj = tmap.get(int(seed))
            if traj is None:
                continue
            try:
                run = simulate(
                    cube,
                    traj,
                    sched,
                    envmap,
                    spec,
                    plan.promotion_rule,
                    gt_mode=plan.gt.mode,
                    canonical_env=plan.gt.canonical_env,
                    conv=cost,
                    n_dev=n_dev,
                    extractor_tags=extractor_tags,
                    proposer_attempts=attempts.get(int(seed)),
                    items=items,
                    gt_items=gt_items,
                )
            except (MissingCell, IndexError) as e:
                if on_missing != "skip":
                    raise
                warnings.warn(f"policy {spec.name} seed {seed} skipped: {e}", stacklevel=2)
                continue
            out.append(run)
    return out


# --------------------------------------------------------------------------- frames

ROUND_COLUMNS: tuple[str, ...] = (
    "seed",
    "policy",
    "policy_kind",
    "round",
    "env_id",
    "ref_id",
    "ref_slot",
    "ref_round",
    "ref_env",
    "ref_age",
    "ref_source",
    "ref_decoding",
    "ref_extractor",
    "ref_env_stale",
    "refresh_kind",
    "wins",
    "losses",
    "ties",
    "n",
    "win_rate",
    "accepted",
    "gt_cand",
    "gt_inc",
    "false_accept",
    "gt_coupled",
    "inc_before",
    "inc_after",
    "cand_slot",
    *(f"calls_{p}" for p in CALL_PURPOSES),
    "calls_round_total",
)

REF_COLUMNS: tuple[str, ...] = (
    "seed",
    "policy",
    "ref_id",
    "slot",
    "created_round",
    "env_id",
    "decoding",
    "extractor_at_storage",
    "source",
    "retired_round",
    "acc_at_storage",
    "rescored_round",
    "parent_id",
)

_CALL_AGGREGATES: tuple[tuple[str, str], ...] = (
    ("calls_candidate_generation", "candidate_generation_calls"),
    ("calls_evaluation", "evaluation_calls"),
    ("calls_reference", "reference_calls"),
    ("calls_total", "total_calls"),
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "policy",
    "label",
    "refresh_rule",
    "kind",
    "adopt_on_promote",
    "n_seeds",
    "gt_acc_mean",
    "gt_acc_sd",
    "gt_acc_final_mean",
    "gt_acc_final_sd",
    "far_seed_mean",
    "far_seed_sd",
    "n_seeds_far",
    "n_evaluated",
    "n_accepted",
    "n_false_accepts",
    "far_pooled",
    "far_lo",
    "far_hi",
    *(c for c, _ in _CALL_AGGREGATES),
    "refreshes",
    "rescores",
    *(f"calls_{p}" for p in CALL_PURPOSES),
    "convention",
    "n_gt_coupled",
    "n_accepted_gt_coupled",
)

PER_SEED_COLUMNS: tuple[str, ...] = (
    "policy",
    "label",
    "seed",
    "gt_acc",
    "gt_acc_final",
    "n_evaluated",
    "n_accepted",
    "n_false_accepts",
    "far",
    "final_inc",
    *(c for c, _ in _CALL_AGGREGATES),
    "refreshes",
    "rescores",
    *(f"calls_{p}" for p in CALL_PURPOSES),
    "n_gt_coupled",
    "n_accepted_gt_coupled",
)

CANDIDATE_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "seed",
    "round",
    "env",
    "incumbent_slot",
    "incumbent_acc",
    "candidate_acc",
    "delta_pp",
    "gt_outcome",
    "dev_delta",
    "advanced",
)

_INT_ROUND_COLS = ("seed", "round", "ref_slot", "ref_round", "ref_age", "wins", "losses", "ties", "n")
_BOOL_ROUND_COLS = ("ref_env_stale", "accepted", "false_accept", "gt_coupled")
_STR_ROUND_COLS = (
    "policy",
    "policy_kind",
    "env_id",
    "ref_id",
    "ref_env",
    "ref_source",
    "ref_decoding",
    "ref_extractor",
    "refresh_kind",
)


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=float) for c in columns})


def runs_to_frames(runs: Sequence[PolicyRun]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(rounds_df, refs_df)``: one row per (seed, policy, round) and per reference snapshot.

    Round calls are flattened into ``calls_<purpose>`` columns (+ ``calls_round_total``). ``retired_round``,
    ``rescored_round`` and ``parent_id`` are nullable (NaN / None) in ``refs_df``.
    """
    rrows: list[dict] = []
    frows: list[dict] = []
    for run in runs:
        for d in run.rounds:
            row = {
                "seed": d.seed,
                "policy": d.policy,
                "policy_kind": run.policy.kind,
                "round": d.round,
                "env_id": d.env_id,
                "ref_id": d.ref_id,
                "ref_slot": d.ref_slot,
                "ref_round": d.ref_round,
                "ref_env": d.ref_env,
                "ref_age": d.ref_age,
                "ref_source": d.ref_source,
                "ref_decoding": d.ref_decoding,
                "ref_extractor": d.ref_extractor,
                "ref_env_stale": d.ref_env_stale,
                "refresh_kind": d.refresh_kind,
                "wins": d.wins,
                "losses": d.losses,
                "ties": d.ties,
                "n": d.n,
                "win_rate": d.wins / d.n if d.n else float("nan"),
                "accepted": d.accepted,
                "gt_cand": d.gt_cand,
                "gt_inc": d.gt_inc,
                "false_accept": d.false_accept,
                "gt_coupled": d.gt_coupled,
                "inc_before": d.inc_before,
                "inc_after": d.inc_after,
                "cand_slot": d.cand_slot,
            }
            for p in CALL_PURPOSES:
                row[f"calls_{p}"] = int(d.calls.get(p, 0))
            row["calls_round_total"] = int(sum(d.calls.values()))
            rrows.append(row)
        for s in run.refs:
            frows.append(
                {
                    "seed": run.seed,
                    "policy": run.policy.name,
                    "ref_id": s.ref_id,
                    "slot": s.slot,
                    "created_round": s.created_round,
                    "env_id": s.env_id,
                    "decoding": s.decoding,
                    "extractor_at_storage": s.extractor_at_storage,
                    "source": s.source,
                    "retired_round": s.retired_round,
                    "acc_at_storage": s.acc_at_storage,
                    "rescored_round": s.rescored_round,
                    "parent_id": s.parent_id,
                }
            )
    rounds_df = pd.DataFrame(rrows, columns=list(ROUND_COLUMNS)) if rrows else _empty(ROUND_COLUMNS)
    refs_df = pd.DataFrame(frows, columns=list(REF_COLUMNS)) if frows else _empty(REF_COLUMNS)
    if not rrows:
        for c in _INT_ROUND_COLS:
            rounds_df[c] = rounds_df[c].astype(np.int64)
        for c in _BOOL_ROUND_COLS:
            rounds_df[c] = rounds_df[c].astype(bool)
        for c in _STR_ROUND_COLS:
            rounds_df[c] = rounds_df[c].astype(str)
    return rounds_df, refs_df


def _mean_or_int(values: Sequence[float]) -> float | int:
    """Mean over seeds; an int when every seed has the same integral value."""
    vals = [float(v) for v in values]
    if not vals:
        return float("nan")
    if all(v == vals[0] for v in vals) and float(vals[0]).is_integer():
        return int(vals[0])
    return float(np.mean(vals))


def _int_if_integral(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Cast columns whose every value is a finite integer to int64 (others stay float)."""
    for c in cols:
        if c not in df.columns or df.empty:
            continue
        v = pd.to_numeric(df[c], errors="coerce").astype(float)
        if np.isfinite(v).all() and (v == np.round(v)).all():
            df[c] = v.astype(np.int64)
        else:
            df[c] = v
    return df


def _group_runs(runs: Sequence[PolicyRun]) -> dict[str, list[PolicyRun]]:
    groups: dict[str, list[PolicyRun]] = {}
    for r in runs:
        groups.setdefault(r.policy.name, []).append(r)
    return groups


def summarize_policies(runs: Sequence[PolicyRun]) -> pd.DataFrame:
    """One row per policy (first-appearance order); columns :data:`SUMMARY_COLUMNS`.

    * ``gt_acc_*``: final accuracy of the shadow incumbent under the canonical env (``gt_acc``) and the
      final round's env (``gt_acc_final``), mean ± sd over seeds;
    * ``far_seed_mean/sd``: FAR per seed, then mean ± sd over the seeds with at least one accept (NaN-aware,
      like the teammate's T3); ``n_seeds_far`` counts those seeds;
    * ``far_pooled`` = sum of false accepts / sum of accepts with a Wilson 95% CI (``far_lo``, ``far_hi``);
    * call columns are means over seeds, int64 when every policy's value is integral; ``refreshes`` and
      ``rescores`` likewise;
    * ``n_gt_coupled`` / ``n_accepted_gt_coupled``: rounds / accepts whose decision equals the GT comparison
      by construction (``RoundDecision.gt_coupled``); when ``n_accepted_gt_coupled == n_accepted`` a zero FAR
      is identical by construction and must be footnoted, not reported as a finding.
    """
    rows: list[dict] = []
    for name, rs in _group_runs(runs).items():
        spec = rs[0].policy
        acc_m, acc_sd, _ = mean_sd([r.final_acc_canonical for r in rs])
        accf_m, accf_sd, _ = mean_sd([r.final_acc_final_env for r in rs])
        far_m, far_sd, far_n = mean_sd([r.far for r in rs])
        n_acc = sum(r.n_accepted for r in rs)
        n_fa = sum(r.n_false_accepts for r in rs)
        p, lo, hi = wilson(n_fa, n_acc)
        row: dict[str, object] = {
            "policy": name,
            "label": spec.label,
            "refresh_rule": spec.refresh_rule,
            "kind": spec.kind,
            "adopt_on_promote": bool(spec.adopt_on_promote),
            "n_seeds": len({r.seed for r in rs}),
            "gt_acc_mean": acc_m,
            "gt_acc_sd": acc_sd,
            "gt_acc_final_mean": accf_m,
            "gt_acc_final_sd": accf_sd,
            "far_seed_mean": far_m,
            "far_seed_sd": far_sd,
            "n_seeds_far": int(far_n),
            "n_evaluated": int(sum(r.n_evaluated for r in rs)),
            "n_accepted": int(n_acc),
            "n_false_accepts": int(n_fa),
            "far_pooled": p,
            "far_lo": lo,
            "far_hi": hi,
        }
        for col, attr in _CALL_AGGREGATES:
            row[col] = _mean_or_int([getattr(r, attr) for r in rs])
        row["refreshes"] = _mean_or_int([r.refreshes for r in rs])
        row["rescores"] = _mean_or_int([r.rescores for r in rs])
        for pur in CALL_PURPOSES:
            row[f"calls_{pur}"] = _mean_or_int([r.calls.get(pur, 0) for r in rs])
        row["convention"] = ",".join(sorted({r.convention for r in rs}))
        row["n_gt_coupled"] = int(sum(r.n_gt_coupled for r in rs))
        row["n_accepted_gt_coupled"] = int(sum(r.n_accepted_gt_coupled for r in rs))
        rows.append(row)
    if not rows:
        return _empty(SUMMARY_COLUMNS)
    df = pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
    count_cols = [c for c, _ in _CALL_AGGREGATES] + ["refreshes", "rescores"]
    return _int_if_integral(df, count_cols + [f"calls_{p}" for p in CALL_PURPOSES])


def per_seed_frame(runs: Sequence[PolicyRun]) -> pd.DataFrame:
    """One row per (policy, seed): the T9 per-seed values (FAR NaN when a seed accepted nothing)."""
    rows: list[dict] = []
    for r in runs:
        row: dict[str, object] = {
            "policy": r.policy.name,
            "label": r.policy.label,
            "seed": r.seed,
            "gt_acc": r.final_acc_canonical,
            "gt_acc_final": r.final_acc_final_env,
            "n_evaluated": r.n_evaluated,
            "n_accepted": r.n_accepted,
            "n_false_accepts": r.n_false_accepts,
            "far": r.far,
            "final_inc": r.final_inc,
        }
        for col, attr in _CALL_AGGREGATES:
            row[col] = int(getattr(r, attr))
        row["refreshes"] = int(r.refreshes)
        row["rescores"] = int(r.rescores)
        for pur in CALL_PURPOSES:
            row[f"calls_{pur}"] = int(r.calls.get(pur, 0))
        row["n_gt_coupled"] = r.n_gt_coupled
        row["n_accepted_gt_coupled"] = r.n_accepted_gt_coupled
        rows.append(row)
    return pd.DataFrame(rows, columns=list(PER_SEED_COLUMNS)) if rows else _empty(PER_SEED_COLUMNS)


def _gt_acc(
    cube: Cube, seed: int, env: Environment, slot: int, round_: int, mode: str, items: np.ndarray | None
) -> float:
    return cube.gt_acc(
        seed, env.decoding.id, slot, round_=round_, extractor=env.extractor, mode=mode, items=items
    )


def _run_matches(run: PolicyRun, sched: EnvSchedule, gt_mode: str) -> bool:
    """Whether ``run`` was simulated under ``sched`` (same env at every round) and ``gt_mode``."""
    if str(run.gt_mode) != str(gt_mode):
        return False
    if not run.schedule:  # hand-built run without provenance: trust the caller
        return True
    R = max((d.round for d in run.rounds), default=0)
    return EnvSchedule(dict(run.schedule)).as_list(R) == sched.as_list(R)


def decision_policies(plan) -> list[str]:
    """Policies with a decision column in :func:`candidates_frame`: headline policies + ORACLE."""
    names = [parse_policy(p).name for p in plan.headline_policies]
    return names + ([] if "ORACLE" in names else ["ORACLE"])


def candidates_frame(
    cube: Cube,
    trajs: Mapping[int, Trajectory] | Sequence[Trajectory],
    plan,
    envs: Mapping[str, Environment] | None,
    runs: Sequence[PolicyRun],
    schedule: EnvSchedule | None = None,
    *,
    extractor_tags: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """T5: one row per (seed, candidate round); columns :data:`CANDIDATE_COLUMNS` + one ``accept``/``reject``
    column per :func:`decision_policies` (named by the policy, e.g. ``"P1"``, ``"ORACLE"``).

    ``incumbent_acc`` / ``candidate_acc`` are GT accuracies of the *trajectory* incumbent ``inc_slot[t]`` and
    of slot t under the schedule's env at round t (plan GT mode); ``gt_outcome`` compares them with tolerance
    ``GT_EPS``. Decisions are taken from ``runs`` that were simulated under this schedule and the plan's GT
    mode (first such run per (policy, seed); runs from other schedules, e.g. ablations, are ignored); a
    (policy, seed) without one is simulated here with the plan's settings and ``extractor_tags`` (decisions
    do not depend on costs).
    """
    sched = _as_schedule(schedule if schedule is not None else plan.env_schedule())
    envmap = as_environments(envs)
    tmap = as_trajectories(trajs)
    dec_names = decision_policies(plan)
    decisions: dict[tuple[str, int], dict[int, bool]] = {}
    for r in runs:
        if not _run_matches(r, sched, plan.gt.mode):
            continue
        decisions.setdefault((r.policy.name, int(r.seed)), {d.round: d.accepted for d in r.rounds})
    seeds = [int(s) for s in cube.seeds if int(s) in tmap]
    missing = [p for p in dec_names if any((p, s) not in decisions for s in seeds)]
    if missing:
        extra = simulate_all(
            cube, tmap, plan, envmap, n_dev=0, extractor_tags=extractor_tags, policies=missing, schedule=sched
        )
        for r in extra:
            decisions.setdefault((r.policy.name, int(r.seed)), {d.round: d.accepted for d in r.rounds})

    _, gt_items = item_sets(cube.n_items, plan.gt.mode)
    mode = "independent_draw" if plan.gt.mode == "split_half" else plan.gt.mode
    rows: list[dict] = []
    for seed in cube.seeds:
        traj = tmap.get(int(seed))
        if traj is None:
            continue
        for t in range(1, traj.R + 1):
            env = _env(envmap, sched.env_at(t))
            inc = int(traj.inc_slot[t])
            a_inc = _gt_acc(cube, int(seed), env, inc, t, mode, gt_items)
            a_cand = _gt_acc(cube, int(seed), env, t, t, mode, gt_items)
            delta = a_cand - a_inc
            outcome = "improves" if delta > GT_EPS else ("worse" if delta < -GT_EPS else "ties")
            row: dict[str, object] = {
                "candidate_id": f"s{seed}-r{t}",
                "seed": int(seed),
                "round": t,
                "env": env.id,
                "incumbent_slot": inc,
                "incumbent_acc": a_inc,
                "candidate_acc": a_cand,
                "delta_pp": delta * 100.0,
                "gt_outcome": outcome,
                "dev_delta": float(traj.cand_dev_acc[t]) - float(traj.inc_dev_acc[t]),
                "advanced": bool(traj.advanced[t]),
            }
            for p in dec_names:
                acc = decisions.get((p, int(seed)), {}).get(t)
                row[p] = "" if acc is None else ("accept" if acc else "reject")
            rows.append(row)
    cols = [*CANDIDATE_COLUMNS, *dec_names]
    return pd.DataFrame(rows, columns=cols) if rows else _empty(cols)


__all__ = [
    "CALL_PURPOSES",
    "CANDIDATE_COLUMNS",
    "CONVENTIONS",
    "GT_EPS",
    "NO_REF",
    "PER_SEED_COLUMNS",
    "REF_COLUMNS",
    "REGENERATE_COMPONENTS",
    "ROUND_COLUMNS",
    "SUMMARY_COLUMNS",
    "CostConvention",
    "PolicyRun",
    "PolicySpec",
    "RefSnapshot",
    "RoundDecision",
    "as_environments",
    "candidates_frame",
    "decision_policies",
    "get_convention",
    "item_sets",
    "parse_policy",
    "per_seed_frame",
    "runs_to_frames",
    "simulate",
    "simulate_all",
    "summarize_policies",
]
