"""AnalysisBundle: the persisted output of ``driftlab analyze`` (contract between analysis, reporting and the app).

On disk (``runs/<run>/exports/analysis/``)::

    bundle.json          meta (see ``AnalysisBundle.meta`` keys below) + list of frame names
    <frame>.csv          one CSV per frame (UTF-8, index not written)

Frames (``AnalysisBundle.frames``) and their producers. Column lists marked *required* are relied on by
reporting/tables.py and the dashboard; producers may add more columns.

* ``pairs``            analysis.allpairs.all_pairs(...).df          one row per (seed, env, j, i); see allpairs.py
* ``pair_summary``     allpairs.summarize_pairs(pr, by=("env","age"))
* ``table4``           allpairs.table4_frame(...)                   (env rows at plan.t4.age)
* ``table7``           allpairs.table7_frame(...)                   (ages x unchanged/changed)
* ``policy_rounds``    policies.runs_to_frames(runs)[0]             one row per (seed, policy, round)
* ``policy_refs``      policies.runs_to_frames(runs)[1]             reference snapshots
* ``policy_summary``   policies.summarize_policies(runs)            one row per policy
* ``candidates``       policies.candidates_frame(...)               Table 5 rows
* ``ablations``        ablations.run_ablations(...)                 Table 8 rows
* ``schedule_random``  ablations.schedule_randomization_frame(...)  EXPLORATORY
* ``hypotheses``       analysis.hypotheses.evaluate_hypotheses(...)
    required: hypothesis, contrast, subject, estimate, ci_lo, ci_hi, unit, n, status, note
* ``ledger_summary``   from the store's ledger table
    required: stage, purpose, seed, n_requested, n_executed, n_cache_hits, prompt_tokens, completion_tokens, wall_s
* ``audit``            from audit_results
    required: decoding_id, slot, repeat, pct_text_identical, pct_correct_flip, n
* ``truncation``       truncation rate per cell
    required: seed, decoding, slot, draw_kind, draw, trunc_rate
* ``trajectory``       one row per (seed, round) incl. round 0
    required: seed, round, incumbent_slot, candidate_slot, inc_dev_acc, cand_dev_acc, advanced, n_attempts,
              is_fallback, origin, prompt_hash, prompt_text
* ``environments``     one row per environment (Table 2 source)
    required: env_id, decoding, temperature, extractor, extractor_tag, fingerprint, description,
              changed_vs_storage (comma-separated component names; "" for the storage env)
* ``accuracy``         ground-truth accuracy per (seed, env, slot, round) for the panels/plots
    required: seed, env, slot, round, gt_acc

``meta`` keys (required): run_id, run_dir, config_hash, plan_hash, plan_version, synthetic (bool), created_at,
seeds, R, N, D, schedule (dict round->env; keys are STRINGS after a JSON round-trip), extractor_tags (dict), engine_fp, config (dict), plan (dict),
provenance (dict), driftlab_version, warnings (list[str]), skipped_pairs (int).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

BUNDLE_DIRNAME = "analysis"
BUNDLE_INDEX = "bundle.json"
SYNTHETIC_NOTE = "SYNTHETIC DATA (mock backend) — not experimental results"
BY_CONSTRUCTION_NOTE = (
    "‡ identical by construction: the compared values come from the same physical generation / the same cells, "
    "so a zero here is a property of the measurement design, not a measured effect."
)

FRAME_NAMES: tuple[str, ...] = (
    "pairs",
    "pair_summary",
    "table4",
    "table7",
    "policy_rounds",
    "policy_refs",
    "policy_summary",
    "candidates",
    "ablations",
    "schedule_random",
    "hypotheses",
    "ledger_summary",
    "audit",
    "truncation",
    "trajectory",
    "environments",
    "accuracy",
)


@dataclass
class AnalysisBundle:
    meta: dict[str, Any]
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)

    def frame(self, name: str) -> pd.DataFrame:
        """A frame by name (an empty DataFrame if the producer did not emit it)."""
        return self.frames.get(name, pd.DataFrame())

    @property
    def synthetic(self) -> bool:
        return bool(self.meta.get("synthetic", False))

    @property
    def plan_hash(self) -> str:
        return str(self.meta.get("plan_hash", ""))


def bundle_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "exports" / BUNDLE_DIRNAME


def _json_default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    return str(o)


def save_bundle(bundle: AnalysisBundle, run_dir: str | Path) -> Path:
    """Write every frame as CSV plus ``bundle.json``; returns the bundle directory."""
    out = bundle_dir(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = []
    for name, df in bundle.frames.items():
        if df is None:
            continue
        df.to_csv(out / f"{name}.csv", index=False)
        names.append(name)
    index = {"meta": bundle.meta, "frames": sorted(names)}
    tmp = out / (BUNDLE_INDEX + ".tmp")
    tmp.write_text(json.dumps(index, indent=2, default=_json_default, sort_keys=True))
    tmp.replace(out / BUNDLE_INDEX)
    return out


def load_bundle(run_dir: str | Path) -> AnalysisBundle:
    """Read a bundle written by :func:`save_bundle` (raises FileNotFoundError if absent)."""
    d = bundle_dir(run_dir)
    index = json.loads((d / BUNDLE_INDEX).read_text())
    frames: dict[str, pd.DataFrame] = {}
    for name in index.get("frames", []):
        path = d / f"{name}.csv"
        if not path.exists():
            continue
        try:
            frames[name] = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            frames[name] = pd.DataFrame()
    return AnalysisBundle(meta=index.get("meta", {}), frames=frames)


def has_bundle(run_dir: str | Path) -> bool:
    return (bundle_dir(run_dir) / BUNDLE_INDEX).exists()
