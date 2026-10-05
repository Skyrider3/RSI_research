"""Phase B planning: the measurement matrix and the determinism-audit requests (docs/ARCHITECTURE.md section 4).

Cells are ALWAYS planned for the full triangle ``(slot k, ("round", r))``, ``0 <= k <= r <= R``, for every seed
and every decoding referenced by an environment, so the cube is uniform; the matrix *mode* only decides which
cells are physically distinct generations:

* greedy: request seed ``None``; nonce ``None`` (a cache hit on the creation generation) unless the cell is
  physical: ``physical_greedy_reruns: all`` -> every ``r > k``; ``ages`` -> the all-pairs rule over BOTH
  reference modes (:func:`physical_greedy_cells`); ``none`` -> never. Physical reruns carry the seed-specific
  nonce ``rerun:s{seed}:{r}``. Creation cells (``r == k``) never get a nonce and are ``physical``.
* sampling (t02) ``full``: seed ``sample_seed(s, "eval", d, k, "round", r, "test", n)`` (an independent draw
  per round, ``physical``); ``lean``: ``draw = -1`` for every round (one shared sample per slot: cache hits
  after the creation cell, ``physical`` only at ``r == k``).
* GT draws (sampling decodings only): ``(k, ("gt", g))`` for ``g < gt_draws`` with
  ``sample_seed(s, "gt", d, k, "gt", g, "test", n)``.
* audit (:func:`audit_tasks`, ``include_audit``): seed ``audit.seed``, slots {0, last incumbent}, draws
  ``("audit", a)`` with nonce ``audit:s{seed}:{a}`` for every decoding; sampling requests reuse the creation
  cell's sampling seed. :func:`check_audit_config` refuses an audit seed outside the run seeds and an audited
  decoding outside the matrix (it has no creation cells to compare with).

Only the EVAL split is ever planned here (:class:`~driftlab.data.EvalSplit` type guard). Groups are ordered by
(seed, draw kind [round, gt, audit], draw, decoding); each group is one ``GenerationEngine.run`` call, so the
ledger splits by seed, purpose and round. The count helpers reproduce :func:`driftlab.estimate.count_requests`
(logical cells and expected executed generations) for any set of trajectories.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from driftlab import keys
from driftlab.analysis.cube import DRAW_AUDIT, DRAW_GT, DRAW_ROUND, Trajectory
from driftlab.backends.base import GenRequest
from driftlab.config import AnalysisPlan, ExperimentConfig
from driftlab.data import EVAL_SPLIT, EvalSplit, Item, user_message
from driftlab.engine import CellKey, CellTask
from driftlab.environments import Decoding

PURPOSE_MATRIX = "eval_matrix"
PURPOSE_GT = "gt_draw"
PURPOSE_AUDIT = "audit"
DRAW_ORDER: dict[str, int] = {DRAW_ROUND: 0, DRAW_GT: 1, DRAW_AUDIT: 2}
LEAN_DRAW = -1  # sampling-seed draw index shared by every round of a slot in lean mode

# (seed, split, decoding_id, slot, draw_kind, draw, item_idx): the cells-table key without run_id
CellTuple = tuple[int, str, str, int, str, int, int]


@dataclass
class MatrixGroup:
    """One engine call: every planned cell of one (seed, decoding, draw)."""

    seed: int
    decoding_id: str
    draw_kind: str
    draw: int
    purpose: str
    tasks: list[CellTask] = field(default_factory=list)

    @property
    def round(self) -> int | None:
        """Ledger ``round`` of the group (the draw for round draws, ``None`` otherwise)."""
        return self.draw if self.draw_kind == DRAW_ROUND else None

    def sort_key(self) -> tuple[int, int, int, str]:
        return (self.seed, DRAW_ORDER[self.draw_kind], self.draw, self.decoding_id)

    def __len__(self) -> int:
        return len(self.tasks)


# --------------------------------------------------------------------------- small helpers


def cell_tuple(cell: CellKey) -> CellTuple:
    """The ``Store.existing_cell_keys`` tuple of a cell."""
    return (cell.seed, cell.split, cell.decoding_id, cell.slot, cell.draw_kind, cell.draw, cell.item_idx)


def matrix_decodings(cfg: ExperimentConfig) -> list[str]:
    """Decodings referenced by the configured environments, in ``cfg.decodings`` order."""
    used = {e.decoding for e in cfg.environments.values()}
    return [d for d in cfg.decodings if d in used]


def sampling_decodings(cfg: ExperimentConfig) -> list[str]:
    """The non-greedy :func:`matrix_decodings` (the ones that get GT draws)."""
    return [d for d in matrix_decodings(cfg) if not cfg.decoding(d).is_greedy]


def rerun_nonce(seed: int, round_: int) -> str:
    """Nonce of a physical greedy rerun (seed-specific: slot 0 is the same prompt in every seed)."""
    return f"rerun:s{seed}:{round_}"


def audit_nonce(seed: int, repeat: int) -> str:
    return f"audit:s{seed}:{repeat}"


def eval_seed(
    cfg: ExperimentConfig, seed: int, decoding_id: str, slot: int, round_: int, item_idx: int
) -> int:
    """Sampling seed of the eval cell ``(slot, ("round", round_))`` (``draw = -1`` in lean mode)."""
    draw = LEAN_DRAW if cfg.matrix.mode == "lean" else round_
    return keys.sample_seed(seed, "eval", decoding_id, slot, DRAW_ROUND, draw, EVAL_SPLIT, item_idx)


def gt_seed(seed: int, decoding_id: str, slot: int, draw: int, item_idx: int) -> int:
    """Sampling seed of the ground-truth cell ``(slot, ("gt", draw))``."""
    return keys.sample_seed(seed, "gt", decoding_id, slot, DRAW_GT, draw, EVAL_SPLIT, item_idx)


def _eval_items(eval_items: EvalSplit) -> list[Item]:
    if not isinstance(eval_items, EvalSplit):
        raise TypeError(
            f"Phase B plans the EVAL split only: expected an EvalSplit, got {type(eval_items).__name__}"
        )
    return list(eval_items)


def _check_traj(cfg: ExperimentConfig, traj: Trajectory) -> None:
    R = cfg.run.rounds
    if len(traj.prompts) != R + 1 or len(traj.inc_slot) != R + 1:
        raise ValueError(
            f"trajectory of seed {traj.seed} has {len(traj.prompts)} slots / {len(traj.inc_slot)} inc_slot "
            f"entries; the config has R = {R} rounds (R + 1 slots)"
        )


def physical_greedy_cells(cfg: ExperimentConfig, inc_slot: Sequence[int]) -> set[tuple[int, int]]:
    """Greedy rerun cells ``(slot k, round r)``, ``r > k``, that get a nonce (creation cells excluded).

    ``ages``: cell (k, r) is physical iff some all-pairs pair (i, j = r), ``j - i`` in ``physical_ages``, has
    ``k == inc_slot[i]`` (incumbent reference mode) or ``k == i`` (chain mode); both modes are planned.
    """
    R = cfg.run.rounds
    if len(inc_slot) != R + 1:
        raise ValueError(f"inc_slot must have R + 1 = {R + 1} entries, got {len(inc_slot)}")
    mode = cfg.matrix.physical_greedy_reruns
    if mode == "none":
        return set()
    if mode == "all":
        return {(k, r) for r in range(1, R + 1) for k in range(r)}
    ages = sorted({int(a) for a in cfg.matrix.physical_ages if 0 <= int(a) <= R})
    out: set[tuple[int, int]] = set()
    for j in range(1, R + 1):
        for a in ages:
            i = j - a
            if i < 0:
                continue
            for k in (int(inc_slot[i]), i):
                if k < j:
                    out.add((k, j))
    return out


# --------------------------------------------------------------------------- matrix


def plan_matrix(
    cfg: ExperimentConfig,
    plan: AnalysisPlan | None,
    trajs: Mapping[int, Trajectory],
    eval_items: EvalSplit,
    *,
    include_audit: bool = False,
    run_id: str | None = None,
) -> list[MatrixGroup]:
    """Every Phase B cell as engine-ready groups ordered by (seed, draw kind, draw, decoding).

    ``plan`` is accepted for the record only: the planner always covers both reference modes. ``run_id``
    defaults to ``cfg.run.name``. ``include_audit`` appends the audit groups (:func:`audit_groups`) when the
    audit is enabled (the audit stage itself runs them shuffled, see :mod:`driftlab.audit`).
    """
    del plan  # both reference modes are planned regardless of plan.reference_mode
    items = _eval_items(eval_items)
    if include_audit:
        check_audit_config(cfg)
    run_id = run_id or cfg.run.name
    R = cfg.run.rounds
    lean = cfg.matrix.mode == "lean"
    decs: dict[str, Decoding] = {d: cfg.decoding(d) for d in matrix_decodings(cfg)}
    users = [user_message(cfg, it.question) for it in items]
    groups: list[MatrixGroup] = []
    for seed in cfg.run.seeds:
        if seed not in trajs:
            raise ValueError(f"no trajectory for seed {seed}")
        traj = trajs[seed]
        _check_traj(cfg, traj)
        phys = physical_greedy_cells(cfg, traj.inc_slot)
        for r in range(R + 1):
            for d, dec in decs.items():
                tasks: list[CellTask] = []
                for k in range(r + 1):
                    prompt = traj.prompts[k]
                    creation = k == r
                    if dec.is_greedy:
                        nonced = not creation and (k, r) in phys
                        nonce = rerun_nonce(seed, r) if nonced else None
                        physical = creation or nonced
                    else:
                        nonce, physical = None, creation or not lean
                    for it, user in zip(items, users, strict=True):
                        req_seed = None if dec.is_greedy else eval_seed(cfg, seed, d, k, r, it.idx)
                        tasks.append(
                            CellTask(
                                CellKey(run_id, seed, EVAL_SPLIT, d, k, DRAW_ROUND, r, it.idx),
                                GenRequest(prompt, user, dec, seed=req_seed, nonce=nonce),
                                physical,
                            )
                        )
                groups.append(MatrixGroup(seed, d, DRAW_ROUND, r, PURPOSE_MATRIX, tasks))
        for g in range(cfg.matrix.gt_draws):
            for d, dec in decs.items():
                if dec.is_greedy:
                    continue
                tasks = [
                    CellTask(
                        CellKey(run_id, seed, EVAL_SPLIT, d, k, DRAW_GT, g, it.idx),
                        GenRequest(traj.prompts[k], user, dec, seed=gt_seed(seed, d, k, g, it.idx)),
                        True,
                    )
                    for k in range(R + 1)
                    for it, user in zip(items, users, strict=True)
                ]
                groups.append(MatrixGroup(seed, d, DRAW_GT, g, PURPOSE_GT, tasks))
        if include_audit and cfg.audit.enabled and seed == cfg.audit.seed:
            groups.extend(audit_groups(cfg, trajs, eval_items, run_id=run_id))
    groups.sort(key=MatrixGroup.sort_key)
    return groups


def creation_tasks(
    cfg: ExperimentConfig,
    traj: Trajectory,
    eval_items: EvalSplit,
    decoding_id: str,
    slot: int,
    *,
    run_id: str | None = None,
) -> list[CellTask]:
    """The creation cell ``(slot, ("round", slot))`` of one decoding, exactly as :func:`plan_matrix` plans it."""
    items = _eval_items(eval_items)
    run_id = run_id or cfg.run.name
    dec = cfg.decoding(decoding_id)
    return [
        CellTask(
            CellKey(run_id, traj.seed, EVAL_SPLIT, decoding_id, slot, DRAW_ROUND, slot, it.idx),
            GenRequest(
                traj.prompts[slot],
                user_message(cfg, it.question),
                dec,
                seed=None if dec.is_greedy else eval_seed(cfg, traj.seed, decoding_id, slot, slot, it.idx),
            ),
            True,
        )
        for it in items
    ]


# --------------------------------------------------------------------------- audit


def audit_slots(cfg: ExperimentConfig, traj: Trajectory) -> list[int]:
    """Audited slots of ``traj`` (``first`` -> 0, ``last_incumbent`` -> incumbent after round R, or an
    explicit slot number), deduplicated in config order."""
    out: list[int] = []
    for name in cfg.audit.slots:
        key = str(name).strip()
        if key == "first":
            k = 0
        elif key in ("last_incumbent", "last"):
            k = traj.incumbent_after(traj.R)
        else:
            try:
                k = int(key)
            except ValueError:
                raise ValueError(
                    f"unknown audit slot {name!r} (use first, last_incumbent or a number)"
                ) from None
            if not 0 <= k <= traj.R:
                raise ValueError(f"audit slot {k} out of range 0..{traj.R}")
        if k not in out:
            out.append(k)
    return out


def check_audit_config(cfg: ExperimentConfig) -> None:
    """``ValueError`` unless ``audit.seed`` is a run seed and every ``audit.decodings`` entry is a decoding of
    the measurement matrix (:func:`matrix_decodings`). An audited decoding outside the matrix has no creation
    cells to compare with: the audit would have to generate them, injecting a decoding that no environment
    uses into the cube. No-op when the audit is disabled."""
    a = cfg.audit
    if not a.enabled:
        return
    if a.seed not in cfg.run.seeds:
        raise ValueError(f"audit.seed {a.seed} is not a run seed {list(cfg.run.seeds)}")
    decs = matrix_decodings(cfg)
    bad = [d for d in a.decodings if d not in decs]
    if bad:
        raise ValueError(
            f"audit.decodings {bad} are not decodings of the measurement matrix {decs} (no creation cells to "
            "compare with); restrict audit.decodings to decodings that some environment uses"
        )


def n_audit_items(cfg: ExperimentConfig) -> int:
    """Number of audited eval items: ``min(audit.n_items, data.eval.n)`` (0 if non-positive)."""
    return max(0, min(int(cfg.audit.n_items), int(cfg.data.eval.n)))


def audit_items(cfg: ExperimentConfig, eval_items: EvalSplit) -> list[Item]:
    """Eval items ``[0, audit.n_items)``."""
    return _eval_items(eval_items)[: max(0, int(cfg.audit.n_items))]


def audit_tasks(
    cfg: ExperimentConfig,
    trajs: Mapping[int, Trajectory],
    eval_items: EvalSplit,
    *,
    run_id: str | None = None,
) -> list[CellTask]:
    """Audit cells in canonical order (slot, decoding, repeat, item); nonce ``audit:s{seed}:{a}`` for every
    decoding, sampling requests reuse the creation cell's sampling seed (:func:`eval_seed` at round = slot).
    ``ValueError`` for an invalid audit config (:func:`check_audit_config`)."""
    run_id = run_id or cfg.run.name
    check_audit_config(cfg)
    seed = cfg.audit.seed
    if seed not in trajs:
        raise ValueError(f"audit.seed {seed} has no trajectory (run seeds: {sorted(trajs)})")
    traj = trajs[seed]
    items = audit_items(cfg, eval_items)
    users = [user_message(cfg, it.question) for it in items]
    tasks: list[CellTask] = []
    for k in audit_slots(cfg, traj):
        for d in cfg.audit.decodings:
            dec = cfg.decoding(d)
            for a in range(cfg.audit.repeats):
                nonce = audit_nonce(seed, a)
                for it, user in zip(items, users, strict=True):
                    req_seed = None if dec.is_greedy else eval_seed(cfg, seed, d, k, k, it.idx)
                    tasks.append(
                        CellTask(
                            CellKey(run_id, seed, EVAL_SPLIT, d, k, DRAW_AUDIT, a, it.idx),
                            GenRequest(traj.prompts[k], user, dec, seed=req_seed, nonce=nonce),
                            True,
                        )
                    )
    return tasks


def audit_groups(
    cfg: ExperimentConfig,
    trajs: Mapping[int, Trajectory],
    eval_items: EvalSplit,
    *,
    run_id: str | None = None,
) -> list[MatrixGroup]:
    """:func:`audit_tasks` grouped by (repeat, decoding) (for counting; the audit stage shuffles them)."""
    by: dict[tuple[int, str], list[CellTask]] = {}
    for t in audit_tasks(cfg, trajs, eval_items, run_id=run_id):
        assert t.cell is not None
        by.setdefault((t.cell.draw, t.cell.decoding_id), []).append(t)
    groups = [
        MatrixGroup(cfg.audit.seed, d, DRAW_AUDIT, a, PURPOSE_AUDIT, tasks) for (a, d), tasks in by.items()
    ]
    groups.sort(key=MatrixGroup.sort_key)
    return groups


# --------------------------------------------------------------------------- counts


def request_identity(req: GenRequest) -> tuple:
    """What determines a request's ``gen_key`` apart from the engine: prompt, user turn, decoding
    parameters, seed (sampling only) and nonce. Equal identities are one physical generation."""
    d = req.decoding
    return (
        req.system,
        req.user,
        tuple(sorted(d.params().items())),
        None if d.is_greedy else req.seed,
        req.nonce,
    )


def count_groups(groups: Iterable[MatrixGroup], known: Iterable[tuple] = ()) -> dict[str, Any]:
    """Logical cells, physical-flagged cells and expected executed generations per purpose.

    Executed = distinct :func:`request_identity` values not in ``known``, attributed to the first group (in
    order) that requests them, exactly as the engine's cache-miss accounting does.
    """
    logical: dict[str, int] = {}
    executed: dict[str, int] = {}
    physical: dict[str, int] = {}
    seen = set(known)
    for g in groups:
        p = g.purpose
        for counter in (logical, executed, physical):
            counter.setdefault(p, 0)
        logical[p] += len(g.tasks)
        for t in g.tasks:
            physical[p] += int(t.physical)
            ident = request_identity(t.request)
            if ident not in seen:
                seen.add(ident)
                executed[p] += 1
    return {
        "logical": logical,
        "executed": executed,
        "physical": physical,
        "logical_total": sum(logical.values()),
        "executed_total": sum(executed.values()),
    }


def count_plan(
    cfg: ExperimentConfig,
    plan: AnalysisPlan | None,
    trajs: Mapping[int, Trajectory],
    eval_items: EvalSplit,
    *,
    include_audit: bool = True,
) -> dict[str, Any]:
    """:func:`count_groups` of :func:`plan_matrix` (same keys; comparable with ``estimate.count_requests``
    ``by_purpose`` for ``eval_matrix`` / ``gt_draw`` / ``audit``)."""
    return count_groups(plan_matrix(cfg, plan, trajs, eval_items, include_audit=include_audit))


def planned_cell_counts(cfg: ExperimentConfig) -> dict[str, int]:
    """Logical matrix cells per purpose (trajectory-independent: the triangle is always full)."""
    S, R, N = len(cfg.run.seeds), cfg.run.rounds, cfg.data.eval.n
    K = R + 1
    T = K * (K + 1) // 2
    return {
        PURPOSE_MATRIX: S * T * N * len(matrix_decodings(cfg)),
        PURPOSE_GT: S * K * N * cfg.matrix.gt_draws * len(sampling_decodings(cfg)),
    }


def planned_cell_keys(cfg: ExperimentConfig, item_idxs: Iterable[int] | None = None) -> set[CellTuple]:
    """Keys of every round / GT matrix cell (trajectory-independent). ``item_idxs`` defaults to
    ``range(data.eval.n)`` (eval items are the first N rows in published order)."""
    idxs = list(range(cfg.data.eval.n) if item_idxs is None else item_idxs)
    R = cfg.run.rounds
    decs = matrix_decodings(cfg)
    samp = set(sampling_decodings(cfg))
    out: set[CellTuple] = set()
    for s in cfg.run.seeds:
        for d in decs:
            for r in range(R + 1):
                for k in range(r + 1):
                    out.update((s, EVAL_SPLIT, d, k, DRAW_ROUND, r, n) for n in idxs)
            if d in samp:
                for g in range(cfg.matrix.gt_draws):
                    for k in range(R + 1):
                        out.update((s, EVAL_SPLIT, d, k, DRAW_GT, g, n) for n in idxs)
    return out


def planned_audit_keys(
    cfg: ExperimentConfig, trajs: Mapping[int, Trajectory], item_idxs: Iterable[int] | None = None
) -> set[CellTuple]:
    """Keys of every audit cell (empty when the audit is disabled)."""
    if not cfg.audit.enabled:
        return set()
    check_audit_config(cfg)
    idxs = list(range(cfg.data.eval.n) if item_idxs is None else item_idxs)[: max(0, int(cfg.audit.n_items))]
    seed = cfg.audit.seed
    if seed not in trajs:
        raise ValueError(f"audit.seed {seed} has no trajectory (run seeds: {sorted(trajs)})")
    return {
        (seed, EVAL_SPLIT, d, k, DRAW_AUDIT, a, n)
        for k in audit_slots(cfg, trajs[seed])
        for d in cfg.audit.decodings
        for a in range(cfg.audit.repeats)
        for n in idxs
    }


__all__ = [
    "DRAW_ORDER",
    "LEAN_DRAW",
    "PURPOSE_AUDIT",
    "PURPOSE_GT",
    "PURPOSE_MATRIX",
    "CellTuple",
    "MatrixGroup",
    "audit_groups",
    "audit_items",
    "audit_nonce",
    "audit_slots",
    "audit_tasks",
    "cell_tuple",
    "check_audit_config",
    "count_groups",
    "count_plan",
    "creation_tasks",
    "eval_seed",
    "gt_seed",
    "matrix_decodings",
    "n_audit_items",
    "physical_greedy_cells",
    "plan_matrix",
    "planned_audit_keys",
    "planned_cell_counts",
    "planned_cell_keys",
    "request_identity",
    "rerun_nonce",
    "sampling_decodings",
]
