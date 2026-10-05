"""Run-cost estimator: request counts per stage / purpose and wall-time ranges per GPU and backend.

Counts follow the planner rules of docs/ARCHITECTURE.md sections 3-4. Notation: S seeds, R rounds
(K = R + 1 slots), N eval items, D dev items, T = K(K+1)/2 triangle cells per (seed, decoding).

* **logical** = cells / requests the pipeline issues; **executed** = cache misses (physical generations
  actually paid for). ``executed`` is the worst case over the trajectories Phase A can produce and
  ``executed_min`` the best case; they differ only for ``physical_greedy_reruns: ages`` (see below).
* Phase A per seed: slot-0 dev D (identical greedy request for every seed -> executed once in total),
  candidate dev R*D, proposer R*expected_attempts.
* Greedy eval: T*N logical cells per seed = creation cells K*N (slot 0 shared across seeds: executed
  N + S*R*N) + rerun cells R(R+1)/2*N, of which the physical (nonce) reruns are executed:
  ``none`` -> 0; ``all`` -> R(R+1)/2*N per seed; ``ages`` -> the exact planner rule
  (:func:`physical_rerun_cells`): cell (k, r) is physical if some all-pairs pair (i, j=r) with
  ``inc_slot[i] == k`` (incumbent mode) or ``i == k`` (chain mode) has ``j - i`` in ``physical_ages``.
  Both reference modes are planned, so the count depends on the trajectory (``inc_slot``): with
  ``inc_slots`` given it is exact; otherwise ``mode: static`` / ``advance_rule: always`` fix the trajectory
  and ``dev_gated`` reports the min / max over every possible set of accepted rounds
  (:func:`ages_rerun_bounds`; R=11, ages [0,1,3,5,10]: 36..54 cells per seed).
* Sampling eval (t02): T*N logical per seed; ``full`` -> all executed; ``lean`` -> K*N executed per seed.
* GT draws (sampling decodings only): K*N*gt_draws per seed. Audit: slots * repeats * n_items * decodings.

Physical greedy reruns carry the seed-specific nonce ``rerun:s{seed}:{r}``, so slot 0 (the same prompt in
every seed) gets an independent rerun per seed and the per-seed count is exact.

Remaining approximations (the ledger reports the truth after a run):
* Candidate prompts are assumed distinct across seeds (no cross-seed cache hits); the audit counts its
  ``last_incumbent`` slot as distinct from slot 0 unless the config fixes the trajectory (``mode: static``
  -> the last incumbent is slot 0), and counts its t02 draws as executed.

Throughput figures (completion tokens/s, aggregate over large batches) are PLANNING ASSUMPTIONS; recalibrate
with :func:`calibrate_from_ledger` on the ledger of a smoke run. Prefill time is not modelled separately.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

from driftlab.config import AnalysisPlan, ExperimentConfig

GPUS: tuple[str, ...] = ("t4", "l4", "a100", "cpu")
BACKENDS: tuple[str, ...] = ("vllm", "hf", "openai_compat", "mock")
COLAB_SESSION_HOURS = 12.0
DEFAULT_EXPECTED_ATTEMPTS = 1.3
# A calibrated throughput is widened to this band (smoke runs use small batches).
CALIBRATION_BAND: tuple[float, float] = (0.8, 1.25)
# Batched engines (vLLM / an OpenAI-compatible vLLM server) are far from saturated below this many
# requests per chunk, so a throughput calibrated on such chunks under-states full-run throughput.
SMALL_CALIBRATION_BATCH = 128

_VLLM = {"t4": (1000.0, 2000.0), "l4": (2000.0, 4000.0), "a100": (6000.0, 12000.0), "cpu": (5.0, 30.0)}
_HF = {"t4": (250.0, 450.0), "l4": (500.0, 900.0), "a100": (1500.0, 2500.0), "cpu": (5.0, 15.0)}
# (backend, gpu) -> (low, high) completion tokens per second. PLANNING ASSUMPTIONS (see module docstring).
# openai_compat assumes a `vllm serve` endpoint on that GPU; the mock backend is effectively free.
THROUGHPUT: dict[tuple[str, str], tuple[float, float]] = {
    **{("vllm", g): v for g, v in _VLLM.items()},
    **{("openai_compat", g): v for g, v in _VLLM.items()},
    **{("hf", g): v for g, v in _HF.items()},
    **{("mock", g): (1e6, 1e7) for g in GPUS},
}
_GPU_ALIASES = {"none": "cpu", "cpu-only": "cpu"}


# --------------------------------------------------------------------------- counts


def _ceil(x: float) -> int:
    return int(math.ceil(round(x, 6)))


def _used_decodings(cfg: ExperimentConfig) -> tuple[list[str], list[str]]:
    """(greedy ids, sampling ids) of the decodings referenced by the configured environments."""
    used = {e.decoding for e in cfg.environments.values()}
    ordered = [d for d in cfg.decodings if d in used]
    greedy = [d for d in ordered if cfg.decodings[d].temperature == 0.0]
    return greedy, [d for d in ordered if d not in greedy]


def _row(
    stage: str,
    purpose: str,
    component: str,
    logical: float,
    executed: float,
    note: str = "",
    executed_min: float | None = None,
) -> dict:
    return {
        "stage": stage,
        "purpose": purpose,
        "component": component,
        "logical": _ceil(logical),
        "executed": _ceil(executed),
        "executed_min": _ceil(executed if executed_min is None else executed_min),
        "note": note,
    }


def _valid_ages(rounds: int, ages: Iterable[int]) -> tuple[int, ...]:
    """Distinct ages that can occur in a pair (0 <= a <= R), sorted."""
    return tuple(sorted({int(a) for a in ages if 0 <= int(a) <= rounds}))


def physical_rerun_cells(rounds: int, ages: Iterable[int], inc_slot: Sequence[int]) -> set[tuple[int, int]]:
    """Greedy rerun cells ``(slot k, round r)`` with ``r > k`` that get a nonce under
    ``physical_greedy_reruns: ages`` (docs/ARCHITECTURE.md section 4).

    Cell (k, r) is physical if some all-pairs pair (i, j=r), ``j`` in 1..R and ``i`` in 0..j, with
    ``inc_slot[i] == k`` (incumbent mode) or ``i == k`` (chain mode) has ``j - i`` in ``ages``.
    ``inc_slot`` is the trajectory incumbent per round 0..R (``inc_slot[0] == inc_slot[1] == 0``).
    """
    R = int(rounds)
    inc = [int(x) for x in inc_slot]
    if len(inc) != R + 1:
        raise ValueError(f"inc_slot must have R + 1 = {R + 1} entries (rounds 0..R), got {len(inc)}")
    if any(not 0 <= k <= max(i - 1, 0) for i, k in enumerate(inc)):
        raise ValueError(f"invalid inc_slot {inc}: need 0 <= inc_slot[i] < i (and inc_slot[0] == 0)")
    out: set[tuple[int, int]] = set()
    for j in range(1, R + 1):
        for a in _valid_ages(R, ages):
            i = j - a
            if i < 0:
                continue
            for k in (inc[i], i):
                if k < j:
                    out.add((k, j))
    return out


@lru_cache(maxsize=128)
def _ages_bounds(rounds: int, ages: tuple[int, ...]) -> tuple[int, int]:
    R, A = rounds, ages
    pos = [a for a in A if a > 0]
    chain = [len({k + a for a in pos if k + a <= R}) for k in range(R + 1)]

    def incumbent_slot(k: int, s: int, e: int) -> int:
        """Cells of slot k when it is the incumbent for rounds i in [s, e] (chain cells included)."""
        js = {k + a for a in pos if k + a <= R}
        js.update(i + a for i in range(s, e + 1) for a in A if 1 <= i + a <= R)
        return len(js)

    # lo/hi[k]: cells of slots >= k given that slot k became the incumbent (slot 0 at round 0, slot k >= 1
    # by being accepted at round k, i.e. it is the incumbent from round k + 1 on).
    lo: dict[int, int] = {}
    hi: dict[int, int] = {}
    for k in range(R - 1, -1, -1):
        s = 0 if k == 0 else k + 1
        options = [(incumbent_slot(k, s, R) + sum(chain[k + 1 :]),) * 2]  # never replaced
        for nxt in range(k + 1, R):  # next accepted slot: k stays incumbent for rounds s..nxt
            base = incumbent_slot(k, s, nxt) + sum(chain[k + 1 : nxt])
            options.append((base + lo[nxt], base + hi[nxt]))
        lo[k] = min(o[0] for o in options)
        hi[k] = max(o[1] for o in options)
    return (lo[0], hi[0]) if R >= 1 else (0, 0)


def ages_rerun_bounds(rounds: int, ages: Iterable[int]) -> tuple[int, int]:
    """(min, max) of ``len(physical_rerun_cells(...))`` over every dev-gated trajectory, i.e. over every set
    of accepted rounds (dynamic programme over the accepted slots; exact for any R)."""
    return _ages_bounds(int(rounds), _valid_ages(rounds, ages))


def _fixed_trajectory(cfg: ExperimentConfig) -> list[int] | None:
    """``inc_slot`` when the config determines it (``mode: static`` or ``advance_rule: always``)."""
    R = cfg.run.rounds
    if cfg.trajectory.mode == "static":
        return [0] * (R + 1)
    if cfg.trajectory.advance_rule == "always":
        return [0] + [max(t - 1, 0) for t in range(1, R + 1)]
    return None


def _inc_slots_for(cfg: ExperimentConfig, inc_slots: Mapping[int, Sequence[int]] | None) -> list[list[int]]:
    seeds = list(cfg.run.seeds)
    missing = [s for s in seeds if s not in (inc_slots or {})]
    if missing:
        raise ValueError(f"inc_slots has no trajectory for seed(s) {missing}")
    return [list((inc_slots or {})[s]) for s in seeds]


def _greedy_rerun_cells(
    cfg: ExperimentConfig, inc_slots: Mapping[int, Sequence[int]] | None
) -> tuple[int, int, str]:
    """(max, min) physical greedy rerun cells summed over seeds (units of N) and a note."""
    R, S, m = cfg.run.rounds, len(cfg.run.seeds), cfg.matrix
    if m.physical_greedy_reruns == "none":
        return 0, 0, "all reruns are cache hits (identical by construction)"
    if m.physical_greedy_reruns == "all":
        return S * R * (R + 1) // 2, S * R * (R + 1) // 2, "physical rerun for every r > k"
    ages = list(_valid_ages(R, m.physical_ages))
    if inc_slots is not None:
        n = sum(len(physical_rerun_cells(R, ages, inc)) for inc in _inc_slots_for(cfg, inc_slots))
        return n, n, f"physical at ages {ages} (exact for the given trajectories)"
    fixed = _fixed_trajectory(cfg)
    if fixed is not None:
        n = S * len(physical_rerun_cells(R, ages, fixed))
        return n, n, f"physical at ages {ages} (trajectory fixed by the config)"
    lo, hi = ages_rerun_bounds(R, ages)
    return S * hi, S * lo, f"physical at ages {ages}: {lo}..{hi} cells/seed depending on the trajectory"


def audit_slot_count(cfg: ExperimentConfig) -> int:
    """Distinct audited slots as far as the config determines them (``planning.audit_slots`` resolves them
    after Phase A): ``first`` and explicit numbers resolve directly; ``last_incumbent`` is slot 0 under
    ``trajectory.mode: static`` and is otherwise counted as a distinct slot (worst case)."""
    resolved: set[int] = set()
    unresolved: set[str] = set()
    for name in cfg.audit.slots:
        key = str(name).strip()
        if key == "first":
            resolved.add(0)
        elif key in ("last_incumbent", "last"):
            if cfg.trajectory.mode == "static":
                resolved.add(0)
            else:
                unresolved.add("last_incumbent")
        else:
            try:
                resolved.add(int(key))
            except ValueError:
                unresolved.add(key)
    return len(resolved) + len(unresolved)


def count_requests(
    cfg: ExperimentConfig,
    plan: AnalysisPlan | None = None,
    expected_attempts: float = DEFAULT_EXPECTED_ATTEMPTS,
    inc_slots: Mapping[int, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Logical and executed request counts (see module docstring for the rules).

    Returns ``{"params", "rows", "by_purpose", "by_stage", "logical_total", "executed_total",
    "executed_min_total", "proposer_max", "notes"}``; each row is ``{stage, purpose, component, logical,
    executed, executed_min, note}`` (``executed`` = worst case over possible trajectories, ``executed_min`` =
    best case). ``inc_slots`` ({seed: inc_slot list for rounds 0..R}, e.g. after Phase A) makes the
    ``ages`` count exact. ``plan`` is only recorded (the planner covers both reference modes regardless).
    """
    R, S = cfg.run.rounds, len(cfg.run.seeds)
    N, D = cfg.data.eval.n, cfg.data.dev.n
    K = R + 1
    T = K * (K + 1) // 2
    reruns_all = R * (R + 1) // 2
    p = cfg.trajectory.proposer
    attempts = min(max(float(expected_attempts), 1.0), float(p.max_attempts))
    m = cfg.matrix
    greedy_ids, sampling_ids = _used_decodings(cfg)
    traj_dec = cfg.environments[cfg.trajectory.env].decoding
    traj_greedy = cfg.decodings[traj_dec].temperature == 0.0

    rows = [
        _row(
            "trajectory",
            "trajectory_dev",
            "slot-0 dev run",
            S * D,
            D if traj_greedy else S * D,
            "identical greedy request for every seed: executed once" if traj_greedy else "",
        ),
        _row("trajectory", "candidate_dev", "candidate dev runs", S * R * D, S * R * D),
        _row(
            "trajectory",
            "proposer",
            "proposer attempts",
            S * R * attempts,
            S * R * attempts,
            f"expected {attempts:g} attempts/round (max {p.max_attempts})",
        ),
    ]
    for g in greedy_ids:
        rows.append(
            _row(
                "matrix",
                "eval_matrix",
                f"{g} creation cells",
                S * K * N,
                N + S * R * N,
                "slot 0 is the same greedy request for every seed",
            )
        )
        phys_hi, phys_lo, note = _greedy_rerun_cells(cfg, inc_slots)
        rows.append(
            _row(
                "matrix",
                "eval_matrix",
                f"{g} rerun cells",
                S * reruns_all * N,
                phys_hi * N,
                note,
                executed_min=phys_lo * N,
            )
        )
    for d in sampling_ids:
        if m.mode == "full":
            rows.append(
                _row("matrix", "eval_matrix", f"{d} triangle", S * T * N, S * T * N, "every round a new draw")
            )
        else:
            rows.append(
                _row(
                    "matrix",
                    "eval_matrix",
                    f"{d} triangle",
                    S * T * N,
                    S * K * N,
                    "lean: one shared draw per slot",
                )
            )
        if m.gt_draws > 0:
            rows.append(
                _row(
                    "matrix",
                    "gt_draw",
                    f"{d} gt draws",
                    S * K * N * m.gt_draws,
                    S * K * N * m.gt_draws,
                    f"{m.gt_draws} independent draw(s) per slot",
                )
            )
    aud = cfg.audit
    if aud.enabled:
        n_slots, n_aud_items = audit_slot_count(cfg), max(0, min(aud.n_items, N))
        n_audit = n_slots * aud.repeats * n_aud_items * len(aud.decodings)
        rows.append(
            _row(
                "audit",
                "audit",
                "determinism audit",
                n_audit,
                n_audit,
                f"seed {aud.seed}: {n_slots} slot(s) x {aud.repeats} repeats x "
                f"{n_aud_items} items x {len(aud.decodings)} decodings",
            )
        )

    by_purpose: dict[str, dict[str, int]] = {}
    by_stage: dict[str, dict[str, int]] = {}
    for r in rows:
        for agg, key in ((by_purpose, r["purpose"]), (by_stage, r["stage"])):
            slot = agg.setdefault(key, {"logical": 0, "executed": 0, "executed_min": 0})
            slot["logical"] += r["logical"]
            slot["executed"] += r["executed"]
            slot["executed_min"] += r["executed_min"]
    notes = [
        "throughput figures are planning assumptions; recalibrate with calibrate_from_ledger",
    ]
    if m.physical_greedy_reruns == "ages" and greedy_ids:
        notes.append(
            "ages: exact planner rule over both reference modes; executed = worst case, executed_min = best "
            "case over the trajectories Phase A can produce (pass inc_slots after Phase A for exact counts)"
        )
    return {
        "params": {
            "run_name": cfg.run.name,
            "config_hash": cfg.config_hash(),
            "plan_hash": plan.plan_hash() if plan is not None else None,
            "reference_mode": plan.reference_mode if plan is not None else None,
            "seeds": S,
            "rounds": R,
            "slots": K,
            "n_eval": N,
            "n_dev": D,
            "matrix_mode": m.mode,
            "physical_greedy_reruns": m.physical_greedy_reruns,
            "physical_ages": list(m.physical_ages),
            "gt_draws": m.gt_draws,
            "greedy_decodings": greedy_ids,
            "sampling_decodings": sampling_ids,
            "expected_attempts": attempts,
            "audit": aud.enabled,
            "smoke": cfg.smoke,
            "trajectory_known": inc_slots is not None,
        },
        "rows": rows,
        "by_purpose": by_purpose,
        "by_stage": by_stage,
        "logical_total": sum(r["logical"] for r in rows),
        "executed_total": sum(r["executed"] for r in rows),
        "executed_min_total": sum(r["executed_min"] for r in rows),
        "proposer_max": S * R * p.max_attempts,
        "notes": notes,
    }


# --------------------------------------------------------------------------- time estimate


@dataclass
class Estimate:
    counts: dict
    executed_total: int
    logical_total: int
    avg_completion_tokens: float
    gpu: str
    backend: str
    hours_lo: float
    hours_hi: float
    warnings: list[str] = field(default_factory=list)
    tokens_per_s: tuple[float, float] = (0.0, 0.0)
    calibrated: bool = False
    executed_min_total: int | None = None  # best case over trajectories (``hours_lo`` uses it)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_GPU_TOKEN_RE = re.compile(r"(?<![a-z0-9])(t4|l4|a100|cpu)(?![a-z0-9])")


def normalize_gpu(gpu: str) -> str:
    """GPU class from a name or an ``nvidia-smi`` string: ``"T4"`` / ``"Tesla T4"`` -> ``"t4"``,
    ``"NVIDIA A100-SXM4-40GB"`` / ``"A100 80GB PCIe"`` -> ``"a100"``; ``ValueError`` for unknown GPUs
    (``"L40S"``, ``"A10G"``, ``"H100"`` are not L4 / A100)."""
    g = gpu.strip().lower()
    g = _GPU_ALIASES.get(g, g)
    m = _GPU_TOKEN_RE.search(g)
    if m:
        return m.group(1)
    raise ValueError(f"unknown gpu {gpu!r}; expected one of {GPUS}")


def normalize_backend(backend: str) -> str:
    b = backend.strip().lower().replace("-", "_")
    if b == "auto":
        return "vllm"
    if b not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS} (or 'auto')")
    return b


def estimate(
    cfg: ExperimentConfig,
    gpu: str = "t4",
    backend: str = "vllm",
    avg_completion_tokens: float = 300.0,
    plan: AnalysisPlan | None = None,
    calibration: dict | None = None,
    inc_slots: Mapping[int, Sequence[int]] | None = None,
) -> Estimate:
    """Hours range for ``cfg`` on ``gpu`` with ``backend``.

    ``hours_lo`` = best-case executed count at the high throughput, ``hours_hi`` = worst-case count at the
    low throughput. ``calibration`` (from :func:`calibrate_from_ledger`) replaces the throughput table by the
    measured tokens/s (widened by :data:`CALIBRATION_BAND`) and, if present, ``avg_completion_tokens`` by the
    measured mean. ``inc_slots`` is passed to :func:`count_requests`.
    """
    g, b = normalize_gpu(gpu), normalize_backend(backend)
    if avg_completion_tokens <= 0:
        raise ValueError("avg_completion_tokens must be > 0")
    counts = count_requests(cfg, plan, inc_slots=inc_slots)
    tps_lo, tps_hi = THROUGHPUT[(b, g)]
    tokens = float(avg_completion_tokens)
    calibrated = False
    if calibration:
        tps = float(calibration["tokens_per_s"])
        if tps <= 0:
            raise ValueError("calibration tokens_per_s must be > 0")
        tps_lo, tps_hi = tps * CALIBRATION_BAND[0], tps * CALIBRATION_BAND[1]
        if calibration.get("mean_completion_tokens"):
            tokens = float(calibration["mean_completion_tokens"])
        calibrated = True
    executed, executed_min = counts["executed_total"], counts["executed_min_total"]
    hours_lo = executed_min * tokens / tps_hi / 3600.0
    hours_hi = executed * tokens / tps_lo / 3600.0

    warnings: list[str] = []
    if hours_hi > COLAB_SESSION_HOURS:
        which = "even the optimistic estimate" if hours_lo > COLAB_SESSION_HOURS else "the upper estimate"
        warnings.append(
            f"{which} ({hours_lo if hours_lo > COLAB_SESSION_HOURS else hours_hi:.1f} h) exceeds a "
            f"{COLAB_SESSION_HOURS:g} h Colab session: the run must resume across sessions "
            "(`driftlab run` is idempotent), or use a faster GPU / vLLM / the lean config"
        )
    if b == "hf" and g == "t4" and cfg.matrix.mode == "full":
        warnings.append("HF transformers on a T4 in full mode is very slow: prefer vLLM or configs/lean.yaml")
    if g == "cpu" and b != "mock":
        warnings.append("CPU inference is only practical for smoke tests (a few items, short outputs)")
    if b == "mock":
        warnings.append("mock backend: SYNTHETIC data; the time estimate is not meaningful")
    if cfg.smoke:
        warnings.append("smoke config: deviates from the protocol (e.g. max_new_tokens != 640)")
    batch = (calibration or {}).get("mean_batch")
    if batch is not None and b in ("vllm", "openai_compat") and float(batch) < SMALL_CALIBRATION_BATCH:
        warnings.append(
            f"throughput calibrated on small chunks ({float(batch):.0f} generations/chunk on average): a "
            f"batched engine runs faster on full-size chunks, so these hours are pessimistic"
        )
    return Estimate(
        counts=counts,
        executed_total=executed,
        logical_total=counts["logical_total"],
        avg_completion_tokens=tokens,
        gpu=g,
        backend=b,
        hours_lo=hours_lo,
        hours_hi=hours_hi,
        warnings=warnings,
        tokens_per_s=(tps_lo, tps_hi),
        calibrated=calibrated,
        executed_min_total=executed_min,
    )


def _fmt_hours(h: float) -> str:
    return f"{h * 60:.0f} min" if h < 1.0 else f"{h:.1f} h"


def format_estimate(e: Estimate) -> str:
    """Aligned plain-text table of the counts, throughput assumption, hours and warnings."""
    p = e.counts["params"]
    header = ("stage", "purpose", "component", "logical", "executed")
    body = [
        (r["stage"], r["purpose"], r["component"], f"{r['logical']:,}", f"{r['executed']:,}")
        for r in e.counts["rows"]
    ]
    total = ("TOTAL", "", "", f"{e.logical_total:,}", f"{e.executed_total:,}")
    widths = [max(len(row[i]) for row in [header, *body, total]) for i in range(len(header))]

    def line(row: Sequence[str]) -> str:
        left = [row[i].ljust(widths[i]) for i in range(3)]
        right = [row[i].rjust(widths[i]) for i in range(3, 5)]
        return "  ".join(left + right).rstrip()

    rule = "  ".join("-" * w for w in widths)
    ages = f" {p['physical_ages']}" if p["physical_greedy_reruns"] == "ages" else ""
    out = [
        f"DriftLab estimate: {p['run_name']} (config {p['config_hash']})",
        f"S={p['seeds']} seeds, R={p['rounds']} rounds, N={p['n_eval']} eval / D={p['n_dev']} dev items; "
        f"matrix {p['matrix_mode']}, greedy reruns {p['physical_greedy_reruns']}{ages}, "
        f"gt_draws {p['gt_draws']}, audit {'on' if p['audit'] else 'off'}",
        "",
        line(header),
        rule,
        *(line(r) for r in body),
        rule,
        line(total),
    ]
    lo = e.executed_min_total
    if lo is not None and lo != e.executed_total:
        out.append(
            f"Executed depends on the trajectory: {lo:,} (best case) - {e.executed_total:,} (worst case, shown)"
        )
    out += [
        "",
        f"Throughput ({e.backend} on {e.gpu}): {e.tokens_per_s[0]:,.0f}-{e.tokens_per_s[1]:,.0f} completion tok/s "
        + (
            "[calibrated from ledger]"
            if e.calibrated
            else "[planning assumption; recalibrate after smoke run]"
        ),
        f"Avg completion tokens: {e.avg_completion_tokens:,.0f} -> "
        f"{e.executed_total * e.avg_completion_tokens / 1e6:,.1f}M tokens executed (worst case)",
        f"Estimated wall time: {_fmt_hours(e.hours_lo)} - {_fmt_hours(e.hours_hi)}",
    ]
    if e.warnings:
        out += ["", "Warnings:", *(f"  ! {w}" for w in e.warnings)]
    return "\n".join(out)


def calibrate_from_ledger(
    ledger_rows: list[dict],
    purposes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Measured throughput from ledger rows (``completion_tokens``, ``wall_s``, optional ``n_executed`` and
    ``purpose``; dicts or ``sqlite3.Row``). Rows without executed generations are skipped.

    Returns ``{"tokens_per_s", "mean_completion_tokens", "mean_batch", "n_rows", "n_executed",
    "completion_tokens", "wall_s"}``; ``mean_completion_tokens`` and ``mean_batch`` (executed generations per
    ledger row, i.e. per chunk) are ``None`` when ``n_executed`` is not recorded.
    """
    tokens, wall, executed, n_rows, have_exec = 0, 0.0, 0, 0, True
    for raw in ledger_rows:
        r = raw if isinstance(raw, Mapping) else dict(raw)
        if purposes is not None and r.get("purpose") not in purposes:
            continue
        ct, ws, ne = r.get("completion_tokens"), r.get("wall_s"), r.get("n_executed")
        if ct is None or ws is None or float(ws) <= 0 or (ne is not None and int(ne) <= 0):
            continue
        tokens += int(ct)
        wall += float(ws)
        n_rows += 1
        if ne is None:
            have_exec = False
        else:
            executed += int(ne)
    if n_rows == 0 or tokens <= 0:
        raise ValueError("no ledger rows with executed generations, completion_tokens and wall_s")
    return {
        "tokens_per_s": tokens / wall,
        "mean_completion_tokens": tokens / executed if have_exec and executed else None,
        "mean_batch": executed / n_rows if have_exec and executed else None,
        "n_rows": n_rows,
        "n_executed": executed,
        "completion_tokens": tokens,
        "wall_s": wall,
    }


__all__ = [
    "BACKENDS",
    "COLAB_SESSION_HOURS",
    "GPUS",
    "THROUGHPUT",
    "Estimate",
    "ages_rerun_bounds",
    "audit_slot_count",
    "calibrate_from_ledger",
    "count_requests",
    "estimate",
    "format_estimate",
    "normalize_backend",
    "normalize_gpu",
    "physical_rerun_cells",
]
