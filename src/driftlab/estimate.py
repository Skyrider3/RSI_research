"""Run-cost estimator: request counts per stage / purpose and wall-time ranges per GPU and backend.

Counts follow the planner rules of docs/ARCHITECTURE.md sections 3-4. Notation: S seeds, R rounds
(K = R + 1 slots), N eval items, D dev items, T = K(K+1)/2 triangle cells per (seed, decoding).

* **logical** = cells / requests the pipeline issues; **executed** = expected cache misses (physical
  generations actually paid for).
* Phase A per seed: slot-0 dev D (identical greedy request for every seed -> executed once in total),
  candidate dev R*D, proposer R*expected_attempts.
* Greedy eval: T*N logical cells per seed = creation cells K*N (slot 0 shared across seeds: executed
  N + S*R*N) + rerun cells R(R+1)/2*N, of which the physical (nonce) reruns are executed:
  ``none`` -> 0; ``all`` -> R(R+1)/2*N per seed; ``ages`` -> sum over a in physical_ages, 0 < a <= R, of
  (R+1-a)*N per seed.
* Sampling eval (t02): T*N logical per seed; ``full`` -> all executed; ``lean`` -> K*N executed per seed.
* GT draws (sampling decodings only): K*N*gt_draws per seed. Audit: slots * repeats * n_items * decodings.

Approximations (upper bounds; the ledger reports the truth after a run):
* ``ages`` counts one physical cell per (creation round i, j = i + a) pair, i.e. it assumes distinct
  incumbents. The real set depends on ``inc_slot`` (shared incumbents merge cells; chain-mode cells add some).
* Physical greedy reruns of slot 0 carry the same nonce for every seed and slot 0 is the same prompt, so they
  share a gen_key across seeds; they are counted once per seed here.
* Candidate prompts are assumed distinct across seeds (no cross-seed cache hits).

Throughput figures (completion tokens/s, aggregate over large batches) are PLANNING ASSUMPTIONS; recalibrate
with :func:`calibrate_from_ledger` on the ledger of a smoke run. Prefill time is not modelled separately.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from driftlab.config import AnalysisPlan, ExperimentConfig

GPUS: tuple[str, ...] = ("t4", "l4", "a100", "cpu")
BACKENDS: tuple[str, ...] = ("vllm", "hf", "openai_compat", "mock")
COLAB_SESSION_HOURS = 12.0
DEFAULT_EXPECTED_ATTEMPTS = 1.3
# A calibrated throughput is widened to this band (smoke runs use small batches).
CALIBRATION_BAND: tuple[float, float] = (0.8, 1.25)

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


def _row(stage: str, purpose: str, component: str, logical: float, executed: float, note: str = "") -> dict:
    return {
        "stage": stage,
        "purpose": purpose,
        "component": component,
        "logical": _ceil(logical),
        "executed": _ceil(executed),
        "note": note,
    }


def _rerun_pairs(R: int, ages: Iterable[int]) -> int:
    """Physical greedy rerun cells per seed (in units of N) for ``physical_greedy_reruns: ages``."""
    return sum(R + 1 - a for a in sorted(set(ages)) if 0 < a <= R)


def count_requests(
    cfg: ExperimentConfig,
    plan: AnalysisPlan | None = None,
    expected_attempts: float = DEFAULT_EXPECTED_ATTEMPTS,
) -> dict[str, Any]:
    """Logical and expected executed request counts (see module docstring for the rules).

    Returns ``{"params", "rows", "by_purpose", "by_stage", "logical_total", "executed_total",
    "proposer_max", "notes"}``; each row is ``{stage, purpose, component, logical, executed, note}``.
    ``plan`` is only recorded (the planner covers both reference modes regardless).
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
        if m.physical_greedy_reruns == "all":
            phys, note = reruns_all, "physical rerun for every r > k"
        elif m.physical_greedy_reruns == "ages":
            ages = [a for a in sorted(set(m.physical_ages)) if 0 < a <= R]
            phys, note = _rerun_pairs(R, ages), f"physical at ages {ages} (approx., distinct incumbents)"
        else:
            phys, note = 0, "all reruns are cache hits (identical by construction)"
        rows.append(_row("matrix", "eval_matrix", f"{g} rerun cells", S * reruns_all * N, S * phys * N, note))
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
        n_audit = len(set(aud.slots)) * aud.repeats * min(aud.n_items, N) * len(aud.decodings)
        rows.append(
            _row(
                "audit",
                "audit",
                "determinism audit",
                n_audit,
                n_audit,
                f"seed {aud.seed}: {len(set(aud.slots))} slots x {aud.repeats} repeats x "
                f"{min(aud.n_items, N)} items x {len(aud.decodings)} decodings",
            )
        )

    by_purpose: dict[str, dict[str, int]] = {}
    by_stage: dict[str, dict[str, int]] = {}
    for r in rows:
        for agg, key in ((by_purpose, r["purpose"]), (by_stage, r["stage"])):
            slot = agg.setdefault(key, {"logical": 0, "executed": 0})
            slot["logical"] += r["logical"]
            slot["executed"] += r["executed"]
    notes = [
        "throughput figures are planning assumptions; recalibrate with calibrate_from_ledger",
        "executed counts are upper bounds: slot-0 physical greedy reruns share gen_keys across seeds",
    ]
    if m.physical_greedy_reruns == "ages":
        notes.append("ages: one physical cell per (i, i+a) pair, assuming distinct incumbents")
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
        },
        "rows": rows,
        "by_purpose": by_purpose,
        "by_stage": by_stage,
        "logical_total": sum(r["logical"] for r in rows),
        "executed_total": sum(r["executed"] for r in rows),
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_gpu(gpu: str) -> str:
    """``"T4"`` -> ``"t4"``, ``"A100-80GB"`` -> ``"a100"``; ``ValueError`` for unknown GPUs."""
    g = gpu.strip().lower().replace("nvidia", "").replace(" ", "").strip("-_")
    g = _GPU_ALIASES.get(g, g)
    m = re.match(r"^(t4|l4|a100|cpu)(?:[-_].*)?$", g)
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
) -> Estimate:
    """Hours range for ``cfg`` on ``gpu`` with ``backend``.

    ``calibration`` (from :func:`calibrate_from_ledger`) replaces the throughput table by the measured
    tokens/s (widened by :data:`CALIBRATION_BAND`) and, if present, ``avg_completion_tokens`` by the measured
    mean.
    """
    g, b = normalize_gpu(gpu), normalize_backend(backend)
    counts = count_requests(cfg, plan)
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
    executed = counts["executed_total"]
    total_tokens = executed * tokens
    hours_lo = total_tokens / tps_hi / 3600.0
    hours_hi = total_tokens / tps_lo / 3600.0

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
        "",
        f"Throughput ({e.backend} on {e.gpu}): {e.tokens_per_s[0]:,.0f}-{e.tokens_per_s[1]:,.0f} completion tok/s "
        + (
            "[calibrated from ledger]"
            if e.calibrated
            else "[planning assumption; recalibrate after smoke run]"
        ),
        f"Avg completion tokens: {e.avg_completion_tokens:,.0f} -> "
        f"{e.executed_total * e.avg_completion_tokens / 1e6:,.1f}M tokens executed",
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

    Returns ``{"tokens_per_s", "mean_completion_tokens", "n_rows", "n_executed", "completion_tokens",
    "wall_s"}``; ``mean_completion_tokens`` is ``None`` when ``n_executed`` is not recorded.
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
    "calibrate_from_ledger",
    "count_requests",
    "estimate",
    "format_estimate",
    "normalize_backend",
    "normalize_gpu",
]
