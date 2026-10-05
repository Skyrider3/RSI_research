"""Phase B planning: triangle cells, physical flags, nonces, sampling seeds, GT draws, audit, counts vs estimate."""

from __future__ import annotations

import itertools

import pytest

from driftlab import keys
from driftlab.analysis.cube import Trajectory, make_trajectory
from driftlab.config import REPO_ROOT, ExperimentConfig, load_config, load_plan
from driftlab.data import DevSplit, EvalSplit, Item
from driftlab.estimate import count_requests, physical_rerun_cells
from driftlab.planning import (
    DRAW_ORDER,
    PURPOSE_AUDIT,
    PURPOSE_GT,
    PURPOSE_MATRIX,
    audit_groups,
    audit_slots,
    audit_tasks,
    cell_tuple,
    check_audit_config,
    count_groups,
    count_plan,
    creation_tasks,
    eval_seed,
    gt_seed,
    n_audit_items,
    physical_greedy_cells,
    plan_matrix,
    planned_audit_keys,
    planned_cell_counts,
    planned_cell_keys,
    request_identity,
    rerun_nonce,
)

CONFIGS = REPO_ROOT / "configs"
PLAN = load_plan(REPO_ROOT / "analysis_plans" / "prereg_v1.yaml")
INITIAL = "Initial prompt: solve step by step and put the answer in \\boxed{}."
MODES = list(itertools.product(["full", "lean"], ["none", "ages", "all"]))
GREEDY_ONLY = [
    "environments={E1: {decoding: greedy, extractor: v1}, E3: {decoding: greedy, extractor: v2}}"
]  # no sampling environment


def _cfg(name: str = "smoke_mock", overrides: list[str] | None = None) -> ExperimentConfig:
    return load_config(CONFIGS / f"{name}.yaml", overrides)


def _eval(n: int) -> EvalSplit:
    return EvalSplit(
        tuple(
            Item("test", i, f"Question {i}: what is {i} + 1?", f"#### {i + 1}", str(i + 1)) for i in range(n)
        )
    )


def _trajs(cfg: ExperimentConfig, advance=lambda s, t: (t + s) % 3 == 1) -> dict[int, Trajectory]:
    """Distinct prompts per (seed, slot >= 1); slot 0 shared by all seeds (as in Phase A)."""
    R = cfg.run.rounds
    out = {}
    for s in cfg.run.seeds:
        prompts = [INITIAL] + [
            f"Seed {s} slot {k}: think carefully, answer in \\boxed{{}}." for k in range(1, R + 1)
        ]
        out[s] = make_trajectory(s, prompts, [False] + [advance(s, t) for t in range(1, R + 1)])
    return out


def _cells(groups):
    for g in groups:
        for t in g.tasks:
            yield g, t


def _check_against_estimate(cfg: ExperimentConfig) -> None:
    ev = _eval(cfg.data.eval.n)
    trajs = _trajs(cfg)
    assert trajs[cfg.audit.seed].incumbent_after(cfg.run.rounds) != 0  # estimate assumes two audit slots
    c = count_plan(cfg, PLAN, trajs, ev, include_audit=True)
    est = count_requests(cfg, PLAN, inc_slots={s: t.inc_slot for s, t in trajs.items()})
    for purpose in (PURPOSE_MATRIX, PURPOSE_GT, PURPOSE_AUDIT):
        e = est["by_purpose"][purpose]
        assert c["logical"][purpose] == e["logical"], purpose
        assert c["executed"][purpose] == e["executed"] == e["executed_min"], purpose
    S, R, N = len(cfg.run.seeds), cfg.run.rounds, cfg.data.eval.n
    K = R + 1
    T = K * (K + 1) // 2
    assert c["logical"][PURPOSE_MATRIX] == S * T * N * 2
    assert c["logical"][PURPOSE_GT] == S * K * N * cfg.matrix.gt_draws
    assert planned_cell_counts(cfg) == {
        PURPOSE_MATRIX: S * T * N * 2,
        PURPOSE_GT: S * K * N * cfg.matrix.gt_draws,
    }


# --------------------------------------------------------------------------- counts vs estimate


@pytest.mark.parametrize(("mode", "reruns"), MODES)
def test_smoke_counts_match_estimate(mode: str, reruns: str) -> None:
    _check_against_estimate(
        _cfg("smoke_mock", [f"matrix.mode={mode}", f"matrix.physical_greedy_reruns={reruns}"])
    )


@pytest.mark.parametrize(("mode", "reruns"), MODES)
def test_eleven_round_counts_match_estimate(mode: str, reruns: str) -> None:
    # protocol shape (S=3, R=11, both decodings, gt 1, audit 2 x 2) with few items to stay fast
    _check_against_estimate(
        _cfg(
            "full",
            [
                "data.eval.n=6",
                "audit.n_items=6",
                f"matrix.mode={mode}",
                f"matrix.physical_greedy_reruns={reruns}",
            ],
        )
    )


@pytest.mark.parametrize("name", ["full", "lean"])
def test_full_size_configs_match_estimate(name: str) -> None:
    _check_against_estimate(_cfg(name))


def test_static_and_always_trajectories_match_estimate_without_inc_slots() -> None:
    for override, adv in (
        ("trajectory.mode=static", lambda s, t: False),
        ("trajectory.advance_rule=always", lambda s, t: True),
    ):
        cfg = _cfg("full", ["data.eval.n=4", "audit.enabled=false", override])
        c = count_plan(cfg, PLAN, _trajs(cfg, adv), _eval(4))
        est = count_requests(cfg, PLAN)  # the config fixes the trajectory
        assert c["executed"][PURPOSE_MATRIX] == est["by_purpose"][PURPOSE_MATRIX]["executed"]
        assert PURPOSE_AUDIT not in c["logical"]


def test_ages_rule_matches_estimate_for_every_trajectory() -> None:
    cfg = _cfg("smoke_mock")  # R = 4
    R = cfg.run.rounds
    for pattern in itertools.product([False, True], repeat=R):
        traj = make_trajectory(0, [f"p{k}" for k in range(R + 1)], [False, *pattern])
        got = physical_greedy_cells(cfg, traj.inc_slot)
        assert got == physical_rerun_cells(R, cfg.matrix.physical_ages, traj.inc_slot)
        assert all(k < r for k, r in got)
    with pytest.raises(ValueError, match="R \\+ 1"):
        physical_greedy_cells(cfg, [0, 0])
    assert (
        physical_greedy_cells(_cfg("smoke_mock", ["matrix.physical_greedy_reruns=none"]), [0] * (R + 1))
        == set()
    )
    assert (
        len(physical_greedy_cells(_cfg("smoke_mock", ["matrix.physical_greedy_reruns=all"]), [0] * (R + 1)))
        == 10
    )


def test_count_groups_is_distinct_request_count() -> None:
    cfg = _cfg()
    groups = plan_matrix(cfg, PLAN, _trajs(cfg), _eval(5))
    idents = {request_identity(t.request) for _, t in _cells(groups)}
    c = count_groups(groups)
    assert c["executed_total"] == len(idents)
    assert c["logical_total"] == sum(len(g) for g in groups)
    assert count_groups(groups, known=idents)["executed_total"] == 0


def test_planned_cell_keys_match_plan() -> None:
    cfg = _cfg()
    ev = _eval(cfg.data.eval.n)
    trajs = _trajs(cfg)
    groups = plan_matrix(cfg, PLAN, trajs, ev)
    keys_ = [cell_tuple(t.cell) for _, t in _cells(groups)]
    assert len(keys_) == len(set(keys_))
    assert set(keys_) == planned_cell_keys(cfg)
    with_audit = plan_matrix(cfg, PLAN, trajs, ev, include_audit=True)
    audit = {cell_tuple(t.cell) for g, t in _cells(with_audit) if g.purpose == PURPOSE_AUDIT}
    assert audit == planned_audit_keys(cfg, trajs)
    assert planned_audit_keys(_cfg("smoke_mock", ["audit.enabled=false"]), trajs) == set()


# --------------------------------------------------------------------------- cell rules


@pytest.mark.parametrize(("mode", "reruns"), MODES)
def test_creation_cells_never_nonced_and_physical(mode: str, reruns: str) -> None:
    cfg = _cfg("smoke_mock", [f"matrix.mode={mode}", f"matrix.physical_greedy_reruns={reruns}"])
    trajs = _trajs(cfg)
    groups = plan_matrix(cfg, PLAN, trajs, _eval(3))
    for _g, t in _cells(groups):
        c = t.cell
        assert c.split == "test" and c.run_id == cfg.run.name
        assert t.request.system == trajs[c.seed].prompts[c.slot]
        if c.draw_kind == "round" and c.slot == c.draw:
            assert t.request.nonce is None and t.physical
        if c.decoding_id == "greedy":
            assert t.request.seed is None
            phys = physical_greedy_cells(cfg, trajs[c.seed].inc_slot)
            if c.slot < c.draw:
                nonced = (c.slot, c.draw) in phys
                assert t.physical == nonced
                assert t.request.nonce == (rerun_nonce(c.seed, c.draw) if nonced else None)
            if reruns == "none":
                assert t.request.nonce is None
        else:
            assert t.request.nonce is None and t.request.seed is not None


def test_seed_specific_rerun_nonces() -> None:
    cfg = _cfg("smoke_mock", ["matrix.physical_greedy_reruns=all"])
    groups = plan_matrix(cfg, PLAN, _trajs(cfg), _eval(2))
    slot0 = {}
    for _, t in _cells(groups):
        c = t.cell
        if c.decoding_id == "greedy" and c.draw_kind == "round" and c.slot < c.draw:
            assert t.request.nonce == f"rerun:s{c.seed}:{c.draw}" and t.physical
            if c.slot == 0:
                slot0[(c.seed, c.draw, c.item_idx)] = request_identity(t.request)
    for (s, r, n), ident in slot0.items():
        if s == 0:
            other = slot0[(1, r, n)]
            assert ident != other and ident[:4] == other[:4]  # same prompt/user/decoding, different nonce


def test_lean_sampling_seed_shared_across_rounds() -> None:
    lean_cfg = _cfg("smoke_mock", ["matrix.mode=lean"])
    full_cfg = _cfg("smoke_mock", ["matrix.mode=full"])
    R = lean_cfg.run.rounds
    for cfg, lean in ((lean_cfg, True), (full_cfg, False)):
        seeds: dict[tuple[int, int, int], set[int]] = {}
        for _, t in _cells(plan_matrix(cfg, PLAN, _trajs(cfg), _eval(3))):
            c = t.cell
            if c.decoding_id != "t02" or c.draw_kind != "round":
                continue
            want_draw = -1 if lean else c.draw
            assert t.request.seed == keys.sample_seed(
                c.seed, "eval", "t02", c.slot, "round", want_draw, "test", c.item_idx
            )
            assert t.request.seed == eval_seed(cfg, c.seed, "t02", c.slot, c.draw, c.item_idx)
            assert t.physical == (not lean or c.slot == c.draw)
            seeds.setdefault((c.seed, c.slot, c.item_idx), set()).add(t.request.seed)
        for (_, k, _), vals in seeds.items():
            assert len(vals) == (1 if lean else R - k + 1)


def test_gt_draws_only_for_sampling_decodings() -> None:
    for gt in (0, 1, 2):
        cfg = _cfg("smoke_mock", [f"matrix.gt_draws={gt}"])
        groups = plan_matrix(cfg, PLAN, _trajs(cfg), _eval(3))
        gt_groups = [g for g in groups if g.draw_kind == "gt"]
        assert len(gt_groups) == gt * len(cfg.run.seeds)
        for g in gt_groups:
            assert g.decoding_id == "t02" and g.purpose == PURPOSE_GT and g.round is None
            assert len(g) == (cfg.run.rounds + 1) * 3
            for t in g.tasks:
                c = t.cell
                assert t.physical and t.request.nonce is None
                assert t.request.seed == gt_seed(c.seed, "t02", c.slot, c.draw, c.item_idx)
                assert t.request.seed == keys.sample_seed(
                    c.seed, "gt", "t02", c.slot, "gt", c.draw, "test", c.item_idx
                )
    greedy_only = _cfg("smoke_mock", GREEDY_ONLY)
    groups = plan_matrix(greedy_only, PLAN, _trajs(greedy_only), _eval(2))
    assert {g.decoding_id for g in groups} == {"greedy"} and all(g.draw_kind == "round" for g in groups)


def test_group_order_and_homogeneity() -> None:
    cfg = _cfg()
    groups = plan_matrix(cfg, PLAN, _trajs(cfg), _eval(2), include_audit=True)
    sk = [(g.seed, DRAW_ORDER[g.draw_kind], g.draw, g.decoding_id) for g in groups]
    assert sk == sorted(sk) and len(sk) == len(set(sk))
    R = cfg.run.rounds
    assert [(g.draw_kind, g.draw, g.decoding_id) for g in groups[:4]] == [
        ("round", 0, "greedy"),
        ("round", 0, "t02"),
        ("round", 1, "greedy"),
        ("round", 1, "t02"),
    ]
    for g in groups:
        assert {(t.cell.seed, t.cell.decoding_id, t.cell.draw_kind, t.cell.draw) for t in g.tasks} == {
            (g.seed, g.decoding_id, g.draw_kind, g.draw)
        }
        if g.draw_kind == "round":
            assert g.purpose == PURPOSE_MATRIX and g.round == g.draw
            assert {t.cell.slot for t in g.tasks} == set(range(g.draw + 1))
        elif g.draw_kind == "audit":
            assert g.purpose == PURPOSE_AUDIT and g.seed == cfg.audit.seed
    assert sum(g.draw_kind == "round" for g in groups) == len(cfg.run.seeds) * (R + 1) * 2


def test_type_guards_and_validation() -> None:
    cfg = _cfg()
    trajs = _trajs(cfg)
    dev = DevSplit(tuple(Item("train", i, f"q{i}", "#### 1", "1") for i in range(3)))
    with pytest.raises(TypeError, match="EVAL"):
        plan_matrix(cfg, PLAN, trajs, dev)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        audit_tasks(cfg, trajs, dev)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no trajectory"):
        plan_matrix(cfg, PLAN, {0: trajs[0]}, _eval(2))
    short = {s: make_trajectory(s, ["a", "b"], [False, True]) for s in cfg.run.seeds}
    with pytest.raises(ValueError, match="R = 4"):
        plan_matrix(cfg, PLAN, short, _eval(2))


# --------------------------------------------------------------------------- audit requests


def test_audit_tasks_nonce_and_creation_seed() -> None:
    for mode in ("full", "lean"):
        cfg = _cfg("smoke_mock", [f"matrix.mode={mode}", "audit.repeats=2", "audit.n_items=5"])
        trajs = _trajs(cfg)
        tasks = audit_tasks(cfg, trajs, _eval(8))
        slots = audit_slots(cfg, trajs[cfg.audit.seed])
        assert slots == [0, trajs[0].incumbent_after(cfg.run.rounds)] and slots[1] != 0
        assert len(tasks) == len(slots) * 2 * 2 * 5
        creation = {
            (t.cell.decoding_id, t.cell.slot, t.cell.item_idx): t.request
            for k in slots
            for d in ("greedy", "t02")
            for t in creation_tasks(cfg, trajs[0], _eval(8), d, k)
        }
        for t in tasks:
            c = t.cell
            assert c.draw_kind == "audit" and c.seed == cfg.audit.seed and c.item_idx < 5 and t.physical
            assert t.request.nonce == f"audit:s{c.seed}:{c.draw}"
            base = creation[(c.decoding_id, c.slot, c.item_idx)]
            assert (t.request.system, t.request.user, t.request.seed) == (base.system, base.user, base.seed)
            if c.decoding_id == "t02":
                assert t.request.seed == eval_seed(cfg, c.seed, "t02", c.slot, c.slot, c.item_idx)
        groups = audit_groups(cfg, trajs, _eval(8))
        assert [(g.draw, g.decoding_id) for g in groups] == [
            (0, "greedy"),
            (0, "t02"),
            (1, "greedy"),
            (1, "t02"),
        ]


def test_audit_slots_dedupe_and_names() -> None:
    cfg = _cfg()
    static = _trajs(cfg, lambda s, t: False)
    assert audit_slots(cfg, static[0]) == [0]
    assert len(audit_tasks(cfg, static, _eval(24))) == 1 * 2 * 1 * 24
    explicit = _cfg("smoke_mock", ["audit.slots=[first, '2', last_incumbent, '0']"])
    traj = _trajs(explicit)[0]
    assert traj.incumbent_after(4) == 4
    assert audit_slots(explicit, traj) == [0, 2, 4]
    with pytest.raises(ValueError, match="unknown audit slot"):
        audit_slots(_cfg("smoke_mock", ["audit.slots=[best]"]), traj)
    with pytest.raises(ValueError, match="out of range"):
        audit_slots(_cfg("smoke_mock", ["audit.slots=['9']"]), traj)
    bad = _cfg("smoke_mock", ["audit.seed=7"])
    with pytest.raises(ValueError, match="audit.seed"):
        audit_tasks(bad, _trajs(cfg), _eval(2))
    with pytest.raises(ValueError, match="audit.seed"):
        plan_matrix(bad, PLAN, _trajs(cfg), _eval(2), include_audit=True)
    with pytest.raises(ValueError, match="audit.seed"):
        planned_audit_keys(bad, _trajs(cfg))
    assert plan_matrix(bad, PLAN, _trajs(cfg), _eval(2))  # the matrix itself does not need the audit seed


def test_audit_config_validation() -> None:
    check_audit_config(_cfg())  # default smoke config is valid
    check_audit_config(_cfg("smoke_mock", ["audit.enabled=false", "audit.seed=7"]))  # disabled: no-op
    with pytest.raises(ValueError, match="audit.seed 7 is not a run seed"):
        check_audit_config(_cfg("smoke_mock", ["audit.seed=7"]))
    # an audited decoding no environment uses has no creation cells: the audit would inject it into the cube
    greedy_only = _cfg("smoke_mock", GREEDY_ONLY)
    trajs = _trajs(greedy_only)
    for call in (
        lambda: check_audit_config(greedy_only),
        lambda: audit_tasks(greedy_only, trajs, _eval(3)),
        lambda: planned_audit_keys(greedy_only, trajs),
        lambda: plan_matrix(greedy_only, PLAN, trajs, _eval(3), include_audit=True),
    ):
        with pytest.raises(ValueError, match=r"audit\.decodings \['t02'\]"):
            call()
    ok = _cfg("smoke_mock", [*GREEDY_ONLY, "audit.decodings=[greedy]"])
    assert {t.cell.decoding_id for t in audit_tasks(ok, _trajs(ok), _eval(3))} == {"greedy"}
    assert n_audit_items(_cfg()) == 24
    assert n_audit_items(_cfg("smoke_mock", ["audit.n_items=500"])) == 24  # capped at data.eval.n
    assert n_audit_items(_cfg("smoke_mock", ["audit.n_items=0"])) == 0
    assert n_audit_items(_cfg("smoke_mock", ["audit.n_items=-3"])) == 0
