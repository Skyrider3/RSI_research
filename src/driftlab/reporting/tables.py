"""Paper tables T1-T10 (+ T7b and the teammate's reported T3/T4) built from an :class:`AnalysisBundle`.

Every table comes in two versions:

* ``TableSpec.df`` -- the *paper* version: exactly the column headers and row labels of the proposal;
* ``TableSpec.extended`` -- every diagnostic the bundle has (CIs, pooled counts, by-construction shares, ...).

Rules (docs/ARCHITECTURE.md section 7): missing data renders as ``"—"`` (tables never crash on empty frames,
e.g. a smoke run with R = 4 has no age-5/10 pairs); values that are identical/zero *by construction* carry
``‡`` and the table then ends with :data:`~driftlab.analysis.bundle_io.BY_CONSTRUCTION_NOTE`; every footnote
list ends with the plan hash, config hash and run id, and with
:data:`~driftlab.analysis.bundle_io.SYNTHETIC_NOTE` for synthetic bundles. The plan-hash footnote says
``— plan not frozen`` unless ``meta["plan_lock"]`` shows the plan was locked (``driftlab freeze-plan``) before
the run started (T1 row ``Pre-registration`` gives the details), and a bundle analysed with
``allow_stale_scores`` puts a ``STALE SCORES`` note first in every table. Bootstrap intervals are described as
what they are: items resampled with one shared index vector, seeds and generations held fixed (they do not
express seed-to-seed variation; the mean ± sd across seeds does, over few seeds). The teammate's reported
numbers (``results/reported/teammate.yaml``) only ever appear in the separate ``T3_reported`` / ``T4_reported``
tables; they are never merged into reproduced values.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from driftlab.analysis.bundle_io import BY_CONSTRUCTION_NOTE, SYNTHETIC_NOTE, AnalysisBundle
from driftlab.analysis.metrics import mean_sd
from driftlab.config import REPO_ROOT, AnalysisPlan

DASH = "—"
DAGGER = "‡"
REPORTED_PATH = REPO_ROOT / "results" / "reported" / "teammate.yaml"
REPORTED_SUFFIX = "Reported by teammate (transcribed; not reproduced by this run)"
TABLE_ORDER: tuple[str, ...] = (
    "T1",
    "T2",
    "T3",
    "T3_reported",
    "T4",
    "T4_reported",
    "T5",
    "T6",
    "T7",
    "T7b",
    "T8",
    "T9",
    "T10",
)

TITLES: dict[str, str] = {
    "T1": "Experimental configuration",
    "T2": "Environment conditions",
    "T3": "Main comparison of reference-refresh policies",
    "T4": "Environment-drift inflation",
    "T5": "Ground-truth evaluation of candidate prompts",
    "T6": "False-accept analysis",
    "T7": "Effect of reference age",
    "T7b": "Effect of reference age: full reference-age × environment grid",
    "T8": "Ablation study",
    "T9": "Reproducibility across random seeds",
    "T10": "System demonstration functionality",
}
TITLES["T3_reported"] = f"{TITLES['T3']} — {REPORTED_SUFFIX}"
TITLES["T4_reported"] = f"{TITLES['T4']} — {REPORTED_SUFFIX}"

# Paper versions: exact headers of the proposal's tables.
PAPER_COLUMNS: dict[str, tuple[str, ...]] = {
    "T1": ("Component", "Configuration"),
    "T2": ("Condition ID", "Environment condition", "Reference version", "Current environment", "Purpose"),
    "T3": (
        "Policy",
        "Reference refresh rule",
        "Ground-truth accuracy (%) ↑",
        "False-accept rate (%) ↓",
        "Model calls ↓",
        "Reference refreshes ↓",
    ),
    "T4": (
        "Environment condition",
        "Reference age (rounds)",
        "Stored-reference win rate (%)",
        "Rerun-reference win rate (%)",
        "Drift inflation (percentage points)",
    ),
    "T5": (
        "Candidate ID",
        "Seed",
        "Incumbent accuracy (%)",
        "Candidate accuracy (%)",
        "Accuracy change (pp)",
        "Ground-truth outcome",
    ),
    "T6": (
        "Refresh policy",
        "Candidates evaluated",
        "Candidates accepted",
        "Accepted candidates with no ground-truth improvement",
        "False-accept rate (%)",
    ),
    "T7": (
        "Reference age (optimization rounds)",
        "Environment status",
        "Stored-reference win rate (%)",
        "Rerun-reference win rate (%)",
        "Drift inflation (pp)",
        "False-accept rate (%)",
    ),
    "T7b": (
        "Reference age (rounds)",
        "Environment",
        "Stored-reference win rate (%)",
        "Rerun-reference win rate (%)",
        "Drift inflation (pp) [95% CI]",
        "False-accept rate, stored reference (%)",
        "False-accept rate, rerun reference (%)",
        "False-accept rate, fresh reference (%)",
        "Decision flip rate (%)",
        "Pairs",
        "Accepts (stored)",
    ),
    "T8": (
        "Ablation",
        "Decoding changes enabled",
        "Extraction changes enabled",
        "Reference-age variation enabled",
        "Drift inflation (pp)",
        "False-accept rate (%)",
    ),
    "T9": (
        "Seed",
        "Refresh policy",
        "Ground-truth accuracy (%)",
        "Drift inflation (pp)",
        "False-accept rate (%)",
        "Model calls",
    ),
    "T10": ("Demonstration component", "Input", "System operation", "Output shown to user", "Status"),
}
PAPER_COLUMNS["T3_reported"] = PAPER_COLUMNS["T3"]
PAPER_COLUMNS["T4_reported"] = PAPER_COLUMNS["T4"]

T1_ROWS: tuple[str, ...] = (
    "Target model",
    "Model checkpoint / revision",
    "Dataset and configuration",
    "Development examples",
    "Evaluation examples",
    "Baseline decoding",
    "Alternative decoding",
    "Answer extraction versions",
    "Number of random seeds",
    "Prompt revision strategy",
    "Software / package versions",
    "Pre-registration",
)
PLAN_NOT_FROZEN = " — plan not frozen"
PLAN_LOCK_UNCHECKED = " — plan lock not checked"
STALE_SCORES_PREFIX = "STALE SCORES:"
# Headline policies: (paper label, refresh rule) exactly as in the proposal.
PAPER_POLICIES: dict[str, tuple[str, str]] = {
    "P1": ("Frozen reference", "Never refresh"),
    "P2": ("Per-batch refresh", "Refresh every batch"),
    "P3": ("Environment-triggered refresh", "Refresh after detected environment changes"),
}
ABLATION_LABELS: dict[str, str] = {
    "A1": "Full system",
    "A2": "No decoding changes",
    "A3": "No extraction changes",
    "A4": "Fixed reference age",
}
OUTCOME_LABELS: dict[str, str] = {"improves": "Improvement", "ties": "No change", "worse": "Regression"}
EXTRACTOR_WORDS: dict[str, str] = {"v1": "strict", "v2": "lenient"}
# (component, input, system operation, output shown to user, dashboard page, bundle frames)
T10_COMPONENTS: tuple[tuple[str, str, str, str, str, str], ...] = (
    (
        "Reference manager",
        "Stored reference outputs with their storage-time scores and environment",
        "Versions every reference (initial / refresh / adopt / re-score), tracks its age and retires it",
        "Reference lifecycle per seed and policy; age and environment of the active reference",
        "2_Reference_Manager.py",
        "policy_refs, policy_rounds",
    ),
    (
        "Environment version tracker",
        "Environment schedule, decoding parameters, extractor source hashes",
        "Fingerprints each environment and detects which components changed against the stored reference",
        "Environment timeline with the changed components (decoding / extractor) per round",
        "1_Environment_Tracker.py",
        "environments, meta.schedule",
    ),
    (
        "Reference rerun mechanism",
        "A stored reference prompt and the current environment",
        "Regenerates the reference under the current environment (physical reruns; live rerun of up to 20 "
        "items via the mock or an OpenAI-compatible server)",
        "Stored vs rerun answers and scores side by side, per-item flips and drift inflation",
        "3_Reference_Rerun.py",
        "pairs",
    ),
    (
        "Candidate comparison view",
        "Candidate outputs and a chosen reference (stored / rerun / fresh)",
        "Paired wins / losses / ties and the promotion decision under the pre-registered rule",
        "2×2 paired table, win rate and decision for each reference",
        "4_Candidate_Comparison.py",
        "pairs, candidates",
    ),
    (
        "Ground-truth evaluation panel",
        "Candidate and incumbent outputs on the held-out test questions",
        "Computes ground-truth accuracy (independent draw under sampling)",
        "Incumbent vs candidate accuracy and the ground-truth outcome of every candidate (Table 5)",
        "5_Ground_Truth.py",
        "candidates, accuracy",
    ),
    (
        "False-accept and drift dashboard",
        "Every candidate / reference pair of the factorial analysis",
        "Aggregates drift inflation and false accepts by environment and reference age with bootstrap CIs",
        "Table 7 and the inflation × reference-age heat map",
        "6_Drift_Dashboard.py",
        "table7, pair_summary",
    ),
    (
        "Refresh-policy comparison",
        "The environment schedule and the stored generations",
        "Dry-run simulation of the refresh policies (P1-P5, Oracle) with model-call accounting",
        "Ground-truth accuracy, false-accept rate and model calls per policy (Tables 3 and 6)",
        "7_Refresh_Policies.py",
        "policy_summary, policy_rounds",
    ),
    (
        "Reproducibility / experiment logs",
        "Run provenance, call ledger and determinism audit",
        "Records config / plan hashes, the engine fingerprint, every model call and the audit results",
        "Provenance, per-seed results (Table 9), ledger and audit tables",
        "9_Reproducibility.py",
        "ledger_summary, audit, meta.provenance",
    ),
)


# --------------------------------------------------------------------------- TableSpec


@dataclass
class TableSpec:
    """One paper table: ``df`` is the paper version (exact headers), ``extended`` the full diagnostics.

    ``footnotes`` belong to the paper version; ``extended_footnotes`` (``None`` -> ``footnotes``) to the
    extended one. Both end with the provenance trailer (plan / config hash, run id, ‡ and synthetic notes).
    """

    id: str
    title: str
    df: pd.DataFrame
    extended: pd.DataFrame | None
    footnotes: list[str]
    caption: str = ""
    extended_footnotes: list[str] | None = None
    reported: bool = False
    synthetic: bool = False  # shown next to a synthetic run (CSV files then start with a marker line)

    def frame(self, extended: bool = False) -> pd.DataFrame:
        """The paper frame, or the extended one when requested and available."""
        return self.extended if extended and self.extended is not None else self.df

    def notes(self, extended: bool = False) -> list[str]:
        """Footnotes of the requested version."""
        if extended and self.extended is not None and self.extended_footnotes is not None:
            return self.extended_footnotes
        return self.footnotes

    def has_dagger(self, extended: bool = False) -> bool:
        return _contains(self.frame(extended), DAGGER)

    @property
    def label(self) -> str:
        """``"Table 3"``, ``"Table 7b"``, ``"Table 3 (reported)"``."""
        return table_label(self.id)


def table_label(table_id: str) -> str:
    base = table_id.removeprefix("T").replace("_reported", "")
    return f"Table {base}" + (" (reported)" if table_id.endswith("_reported") else "")


# --------------------------------------------------------------------------- formatting helpers


def _num(v: Any) -> float:
    """Float of a scalar (NaN for None, "", unparsable strings); bools count as 0/1."""
    if v is None:
        return float("nan")
    if isinstance(v, (bool, np.bool_)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in ("true", "false"):
            return 1.0 if s.lower() == "true" else 0.0
        try:
            return float(s)
        except ValueError:
            return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _finite(v: Any) -> bool:
    return math.isfinite(_num(v))


def _truthy(v: Any) -> bool:
    """Booleans that survive a CSV round trip ("True"/"False" strings; NaN / None -> False)."""
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "1.0")
    if v is None:
        return False
    try:
        if pd.isna(v):
            return False
    except (TypeError, ValueError):
        pass
    return bool(v)


def _int(v: Any, default: int = 0) -> int:
    f = _num(v)
    return int(round(f)) if math.isfinite(f) else default


def _text(v: Any, default: str = "") -> str:
    if v is None:
        return default
    try:
        if pd.isna(v):
            return default
    except (TypeError, ValueError):
        pass
    return str(v)


def _clean_zero(s: str) -> str:
    """``"-0.00"`` -> ``"0.00"`` (a rounded negative zero is not a measured sign)."""
    if s.startswith("-") and s.lstrip("-").replace(".", "").replace("0", "") == "":
        return s[1:]
    return s


def fmt_num(value: Any, digits: int = 1, scale: float = 1.0, signed: bool = False) -> str:
    """``value * scale`` with ``digits`` decimals (``"—"`` when missing); ``signed`` adds a ``+``."""
    f = _num(value)
    if not math.isfinite(f):
        return DASH
    out = _clean_zero(f"{f * scale:.{digits}f}")
    if signed and not out.startswith("-") and float(out) != 0.0:
        out = "+" + out
    return out


def fmt_pct(value: Any, digits: int = 1) -> str:
    """A proportion as a percentage (``0.429`` -> ``"42.9"``)."""
    return fmt_num(value, digits, 100.0)


def fmt_int(value: Any, thousands: bool = False) -> str:
    """Rounded integer; thousands separators only when asked (extended tables)."""
    f = _num(value)
    if not math.isfinite(f):
        return DASH
    i = int(round(f))
    return f"{i:,}" if thousands else str(i)


def fmt_count(value: Any, thousands: bool = False) -> str:
    """An integer when integral, else one decimal (means of counts over seeds)."""
    f = _num(value)
    if not math.isfinite(f):
        return DASH
    if abs(f - round(f)) < 1e-9:
        return fmt_int(f, thousands)
    return f"{f:,.1f}" if thousands else f"{f:.1f}"


def mean_sd_str(mean: Any, sd: Any, digits: int = 1, scale: float = 100.0) -> str:
    """``"71.0 ± 2.0"`` from a precomputed mean and sd (``"—"`` when the mean is missing)."""
    m = fmt_num(mean, digits, scale)
    if m == DASH:
        return DASH
    s = fmt_num(sd, digits, scale)
    return f"{m} ± {s}" if s != DASH else m


def pct_mean_sd(values: Iterable[Any], digits: int = 1, scale: float = 100.0) -> str:
    """Mean ± sample sd over the finite values, as percentages (``[0.69, 0.73]`` -> ``"71.0 ± 2.8"``)."""
    m, sd, n = mean_sd(_num(v) for v in values)
    return DASH if n == 0 else mean_sd_str(m, sd, digits, scale)


def pp_mean_sd(values: Iterable[Any], digits: int = 2, scale: float = 100.0) -> str:
    """Mean ± sd of proportions as percentage points with two decimals (``"2.27 ± 0.81"``)."""
    return pct_mean_sd(values, digits, scale)


def ci_str(lo: Any, hi: Any, digits: int = 2, scale: float = 100.0) -> str:
    """``"[lo, hi]"`` (``"—"`` when either bound is missing)."""
    a, b = fmt_num(lo, digits, scale), fmt_num(hi, digits, scale)
    return DASH if DASH in (a, b) else f"[{a}, {b}]"


def with_ci(value: Any, lo: Any, hi: Any, digits: int = 2, scale: float = 100.0) -> str:
    """``"2.27 [1.10, 3.40]"``; just the value when the CI is missing; ``"—"`` without a value."""
    v = fmt_num(value, digits, scale)
    if v == DASH:
        return DASH
    ci = ci_str(lo, hi, digits, scale)
    return v if ci == DASH else f"{v} {ci}"


def frac_str(k: Any, n: Any, thousands: bool = False) -> str:
    """``"3/7"`` (``"—"`` when n is missing)."""
    if not _finite(n):
        return DASH
    return f"{fmt_int(k, thousands)}/{fmt_int(n, thousands)}"


def yes_no(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return DASH
    return "Yes" if _truthy(v) else "No"


def mark(s: str, flag: bool) -> str:
    """Append ``‡`` to a value that is identical by construction (never to a missing value)."""
    return f"{s} {DAGGER}" if flag and s != DASH else s


def _contains(df: pd.DataFrame | None, token: str) -> bool:
    if df is None or df.empty:
        return False
    return any(token in str(v) for v in df.to_numpy().ravel())


def _frame(rows: Sequence[Sequence[Any]], columns: Sequence[str]) -> pd.DataFrame:
    """A string frame with the given columns (one ``"—"`` row when there are no rows)."""
    data = [[_text(v, DASH) or DASH for v in r] for r in rows]
    if not data:
        data = [[DASH] * len(columns)]
    return pd.DataFrame(data, columns=list(columns), dtype=object)


# --------------------------------------------------------------------------- context


def _plan_from_meta(raw: Any) -> AnalysisPlan:
    """The bundle's analysis plan (defaults when ``meta["plan"]`` is missing or not a valid plan)."""
    return _plan_or_none(raw) or AnalysisPlan()


def _plan_or_none(raw: Any) -> AnalysisPlan | None:
    if isinstance(raw, Mapping):
        try:
            return AnalysisPlan.model_validate(dict(raw))
        except Exception:  # an older / hand-built plan dict
            return None
    return None


def _dig(d: Any, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if not isinstance(d, Mapping) or k not in d:
            return default
        d = d[k]
    return d


def is_synthetic(bundle: AnalysisBundle) -> bool:
    """Whether the bundle holds synthetic data (fail-safe).

    True when ``meta["synthetic"]`` says so, and also when the flag is missing or false but the bundle's config
    or provenance names the mock backend. A bundle that lost its flag is never presented, or written, as real
    data.
    """
    meta = bundle.meta or {}
    if _truthy(meta.get("synthetic")):
        return True
    kinds = (_dig(meta, "config", "backend", "kind"), _dig(meta, "provenance", "backend", "kind"))
    if any(str(k).strip().lower() == "mock" for k in kinds if k):
        return True
    return _truthy(_dig(meta, "provenance", "backend", "synthetic", default=False))


def _canonical_policy(name: str) -> str:
    """Canonical policy spelling (``"p3"`` -> ``"P3"``); unknown names pass through unchanged."""
    from driftlab.analysis.policies import parse_policy

    try:
        return parse_policy(str(name)).name
    except ValueError:
        return str(name)


class _Ctx:
    """Bundle accessors shared by the table builders."""

    def __init__(self, bundle: AnalysisBundle, app_dir: Path | None) -> None:
        self.bundle = bundle
        self.meta: dict[str, Any] = dict(bundle.meta or {})
        self.cfg: dict[str, Any] = dict(self.meta.get("config") or {})
        parsed = _plan_or_none(self.meta.get("plan"))
        self.plan_is_default = parsed is None
        self.plan = parsed or AnalysisPlan()
        self.prov: dict[str, Any] = dict(self.meta.get("provenance") or {})
        self.seeds = [int(s) for s in (self.meta.get("seeds") or _dig(self.cfg, "run", "seeds", default=[]))]
        self.S = len(self.seeds)
        self.R = _int(self.meta.get("R", _dig(self.cfg, "run", "rounds")), 0)
        self.N = _int(self.meta.get("N", _dig(self.cfg, "data", "eval", "n")), 0)
        self.D = _int(self.meta.get("D", _dig(self.cfg, "data", "dev", "n")), 0)
        self.tags: dict[str, str] = {
            str(k): str(v) for k, v in (self.meta.get("extractor_tags") or {}).items()
        }
        self.app_dir = Path(app_dir) if app_dir is not None else REPO_ROOT / "app"
        self.headline = list(dict.fromkeys(_canonical_policy(p) for p in self.plan.headline_policies))
        self.synthetic = is_synthetic(bundle)
        observed = self.meta.get("R_observed")
        self.R_observed: dict[str, int] = (
            {str(k): _int(v) for k, v in observed.items()} if isinstance(observed, Mapping) else {}
        )
        # Rounds completed by every seed (a budget-exhausted run is analysed on what it has).
        self.R_done = min(self.R_observed.values(), default=self.R)
        self.envs = self._environments()

    def frame(self, name: str) -> pd.DataFrame:
        df = self.bundle.frame(name)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

    @property
    def run_id(self) -> str:
        return str(self.meta.get("run_id", "?"))

    @property
    def B(self) -> int:
        return _int(self.meta.get("B", self.plan.bootstrap.B), self.plan.bootstrap.B)

    # ---------------------------------------------------------------- environments
    def _environments(self) -> dict[str, dict[str, Any]]:
        """env_id -> {decoding, temperature, top_p, top_k, repetition_penalty, max_new_tokens, extractor, tag,
        fingerprint, description, changed (list)} from the ``environments`` frame (config fallback)."""
        out: dict[str, dict[str, Any]] = {}
        df = self.frame("environments")
        if not df.empty and "env_id" in df.columns:
            for r in df.to_dict("records"):
                ext = _text(r.get("extractor"))
                changed = [c for c in _text(r.get("changed_vs_storage")).split(",") if c.strip()]
                out[_text(r.get("env_id"))] = {
                    "decoding": _text(r.get("decoding")),
                    "temperature": _num(r.get("temperature")),
                    "top_p": _num(r.get("top_p")),
                    "top_k": _num(r.get("top_k")),
                    "repetition_penalty": _num(r.get("repetition_penalty")),
                    "max_new_tokens": _num(r.get("max_new_tokens")),
                    "extractor": ext,
                    "tag": _text(r.get("extractor_tag")) or self.tags.get(ext, ext),
                    "fingerprint": _text(r.get("fingerprint")),
                    "description": _text(r.get("description")),
                    "changed": [c.strip() for c in changed],
                }
            return out
        decs = self.cfg.get("decodings") or {}
        envs = self.cfg.get("environments") or {}
        max_new = _num(_dig(self.cfg, "model", "max_new_tokens"))
        for eid, e in envs.items():
            d = decs.get(e.get("decoding"), {}) if isinstance(e, Mapping) else {}
            ext = _text(e.get("extractor")) if isinstance(e, Mapping) else ""
            out[str(eid)] = {
                "decoding": _text(e.get("decoding")) if isinstance(e, Mapping) else "",
                "temperature": _num(d.get("temperature")),
                "top_p": _num(d.get("top_p", 1.0)),
                "top_k": _num(d.get("top_k", 0)),
                "repetition_penalty": _num(d.get("repetition_penalty", 1.0)),
                "max_new_tokens": max_new,
                "extractor": ext,
                "tag": self.tags.get(ext, ext),
                "fingerprint": "",
                "description": _text(e.get("description")) if isinstance(e, Mapping) else "",
                "changed": [],
            }
        st = out.get(str(self.plan.storage_env))
        if st is not None:
            for e in out.values():
                ch = []
                if (e["decoding"], e["temperature"]) != (st["decoding"], st["temperature"]):
                    ch.append("decoding")
                if e["extractor"] != st["extractor"]:
                    ch.append("extractor")
                e["changed"] = ch
        return out

    def env_ids(self) -> list[str]:
        return list(self.envs) or ["E1", "E2", "E3", "E4"]

    def decoding_short(self, env_id: str) -> str:
        e = self.envs.get(env_id)
        if e is None:
            return "?"
        t = e["temperature"]
        if math.isfinite(t) and t == 0.0:
            return "greedy"
        return f"T={t:g}" if math.isfinite(t) else e["decoding"] or "?"

    def extractor_short(self, env_id: str) -> str:
        e = self.envs.get(env_id)
        if e is None:
            return "?"
        word = EXTRACTOR_WORDS.get(e["extractor"])
        return f"{word} {e['tag']}" if word else e["tag"]

    def env_short(self, env_id: str) -> str:
        """``"greedy + strict v1@05dd25e6"``."""
        return f"{self.decoding_short(env_id)} + {self.extractor_short(env_id)}"

    def env_label(self, env_id: str) -> str:
        """Paper label of an environment from its changed components (``Unchanged environment`` ...)."""
        e = self.envs.get(env_id)
        if e is None:
            return env_id
        changed = e["changed"]
        if not changed:
            return "Unchanged environment"
        if len(changed) > 1:
            return "Multiple environment changes"
        return {"decoding": "Decoding change", "extractor": "Answer-extraction change"}.get(
            changed[0], f"{changed[0].replace('_', ' ').capitalize()} change"
        )

    def env_purpose(self, env_id: str) -> str:
        changed = (self.envs.get(env_id) or {}).get("changed", [])
        if not changed:
            return "Control: measures the null (no drift)"
        if len(changed) > 1:
            return "Combined drift"
        return {"decoding": "Isolates decoding drift", "extractor": "Isolates scoring drift"}.get(
            changed[0], f"Isolates {changed[0].replace('_', ' ')} drift"
        )

    # ---------------------------------------------------------------- integrity notes
    @property
    def plan_lock(self) -> dict[str, Any] | None:
        """``meta["plan_lock"]`` (None for a bundle written before the pre-registration check)."""
        lock = self.meta.get("plan_lock")
        return dict(lock) if isinstance(lock, Mapping) and lock.get("status") else None

    def plan_hash_suffix(self) -> str:
        """Appended to every footnote that carries the plan hash: ``" — plan not frozen"`` unless the plan was
        locked before the run started (``" — plan lock not checked"`` for a bundle without the check)."""
        lock = self.plan_lock
        if lock is None:
            return PLAN_LOCK_UNCHECKED
        return "" if lock["status"] == "locked_before_run" else PLAN_NOT_FROZEN

    def stale_note(self, reported: bool = False) -> list[str]:
        """The prominent first footnote of every table of a bundle analysed with stale scores."""
        stale = self.meta.get("stale_scores")
        rows = [s for s in stale if isinstance(s, Mapping)] if isinstance(stale, list) else []
        if not rows:
            return []
        old = ", ".join(
            f"{_text(s.get('extractor'))}@{_text(s.get('ext_hash'))} ({_int(s.get('n')):,} score rows)"
            for s in rows
        )
        cur = ", ".join(
            dict.fromkeys(f"{_text(s.get('extractor'))}@{_text(s.get('current_hash'))}" for s in rows)
        )
        note = (
            f"{STALE_SCORES_PREFIX} computed with extractor hashes {old} that differ from the current {cur}: values "
            "that use these extractors may not match the extractor code named in this table. Re-score "
            "(`driftlab run -c <config> --run-dir <dir> --stages score`) and re-run the analysis before reporting."
        )
        if reported:
            note += " (Applies to this run's reproduced tables, not to the transcribed numbers.)"
        return [note]

    def bootstrap_text(self) -> str:
        """What the paired item bootstrap resamples (items only: seeds and generations are held fixed)."""
        what = "decision test questions" if self.plan.gt.mode == "split_half" else "test questions"
        return (
            f"95% paired item bootstrap (B = {self.B:,}, the {self.n_decision} {what} resampled with one shared "
            "index vector; seeds and generations held fixed)"
        )

    def seed_sd_text(self) -> str:
        seeds = "seed" if self.S == 1 else "seeds"
        return (
            f"Mean ± sd across seeds reflects trajectory-to-trajectory variation over only {self.S} {seeds}."
        )

    # ---------------------------------------------------------------- footnote trailer
    def trailer(self, has_dagger: bool, reported: bool = False) -> list[str]:
        ph, ch = self.bundle.plan_hash or "?", str(self.meta.get("config_hash", "?"))
        version = self.meta.get("plan_version", self.plan.version)
        nf = self.plan_hash_suffix()
        if reported:
            prov = (
                f"Shown alongside run {self.run_id} (analysis plan {version}, plan hash {ph}{nf}, config hash "
                f"{ch}); these transcribed numbers do not come from that run."
            )
        else:
            prov = f"Analysis plan {version} (plan hash {ph}){nf}; config hash {ch}; run {self.run_id}."
        out = [prov]
        if has_dagger:
            out.append(BY_CONSTRUCTION_NOTE)
        if self.synthetic:
            out.append(
                f"{SYNTHETIC_NOTE} (applies to this run's reproduced tables, not to the transcribed numbers)"
                if reported
                else SYNTHETIC_NOTE
            )
        return out

    def validity_notes(self) -> list[str]:
        """Notes every reproduced table carries before its provenance line (plan fallback, partial run)."""
        out = []
        if self.plan_is_default:
            out.append(
                "The bundle carries no valid analysis plan; table settings (ages, policies, environments) fall "
                "back to the plan defaults and may not match the run."
            )
        partial = {s: r for s, r in self.R_observed.items() if r < self.R}
        if partial:
            done = ", ".join(f"seed {s}: {r}" for s, r in sorted(partial.items()))
            out.append(
                f"Partial run: completed rounds ({done}) < R = {self.R}; every value covers the completed rounds "
                "only."
            )
        return out

    @property
    def n_decision(self) -> int:
        """Test questions a decision compares on (the first half under ``split_half`` ground truth)."""
        return self.N // 2 if self.plan.gt.mode == "split_half" else self.N

    def skipped_note(self) -> list[str]:
        k = _int(self.meta.get("skipped_pairs"))
        if k <= 0:
            return []
        return [
            f"{k:,} candidate/reference pairs were skipped because a cell they need is missing from the store "
            "(see the bundle warnings)."
        ]

    def far_notes(self, policies: Sequence[str], seed_means: bool = True) -> list[str]:
        """FAR caveats for ``policies``: accepts that cannot be false by construction (only *some* of them:
        a rate made only of such accepts carries ‡) and seed means over fewer seeds than the run has."""
        summ = self.summary_rows()
        part, few = [], []
        for p in policies:
            r = summ.get(p)
            if r is None:
                continue
            n_acc, k = _int(r.get("n_accepted")), _int(r.get("n_accepted_gt_coupled"))
            if 0 < k < n_acc:
                part.append(f"{p} {k} of {n_acc}")
            n_far, n_seeds = _int(r.get("n_seeds_far"), -1), _int(r.get("n_seeds"), -1)
            if 0 < n_far < n_seeds:
                few.append(f"{p} {n_far} of {n_seeds} seeds")
        out = []
        if part:
            out.append(
                "Partly zero by construction: an accept decided against the incumbent's own ground-truth "
                "generations (greedy decoding, the reference is the current incumbent's output under the current "
                "extractor) cannot be a false accept. Accepts of that kind (pooled over seeds): "
                + "; ".join(part)
                + ". They lower these false-accept rates by construction."
            )
        if few and seed_means:
            out.append(
                "False-accept rate averaged over fewer seeds than the run has, because a seed without an accepted "
                "candidate has no rate: " + "; ".join(few) + " (a single seed shows ± 0.0)."
            )
        return out

    # ---------------------------------------------------------------- shared texts
    def schedule_text(self, schedule: Mapping[Any, Any] | None = None) -> str:
        sched = {int(k): str(v) for k, v in (schedule or self.plan.schedule).items()}
        starts = sorted(sched)
        parts = []
        for i, s in enumerate(starts):
            env = sched[s]
            if self.R and s > self.R:
                parts.append(f"{env} from round {s} (not reached with R = {self.R})")
                continue
            end = starts[i + 1] - 1 if i + 1 < len(starts) else self.R
            end = min(end, self.R) if self.R else end
            parts.append(f"rounds {s}–{end}: {env}" if end > s else f"round {s}: {env}")
        return "; ".join(parts)

    def rule_text(self) -> str:
        r = self.plan.promotion_rule
        if r.kind == "net_win":
            return f"accept iff (wins − losses) / n > {r.margin:g}"
        if r.kind == "win_rate":
            return f"accept iff wins / n ≥ {r.tau:g} and wins > losses"
        return f"accept iff wins > losses and the one-sided exact McNemar p < {r.alpha:g}"

    def convention_text(self, conv: str | None = None) -> str:
        conv = conv or self.plan.cost_convention
        base = (
            f"per round {self.D} incumbent dev calls + 1 proposer call + {self.N} candidate evaluations, "
            f"plus {self.N} per reference refresh (re-scoring stored text needs no model call)"
        )
        if conv == "full":
            base += "; also counts candidate dev runs, every proposer attempt and the initial reference"
        return f"Model calls are logical counts under the {conv} convention: {base}."

    def policy_label(self, policy: str, summary_row: Mapping[str, Any] | None = None) -> str:
        if policy in PAPER_POLICIES:
            return PAPER_POLICIES[policy][0]
        return _text((summary_row or {}).get("label")) or policy

    def policy_rule(self, policy: str, summary_row: Mapping[str, Any] | None = None) -> str:
        if policy in PAPER_POLICIES:
            return PAPER_POLICIES[policy][1]
        return _text((summary_row or {}).get("refresh_rule")) or DASH

    def summary_rows(self) -> dict[str, dict[str, Any]]:
        df = self.frame("policy_summary")
        if df.empty or "policy" not in df.columns:
            return {}
        return {_text(r.get("policy")): r for r in df.to_dict("records")}

    def ledger_executed(self) -> tuple[int | None, str]:
        df = self.frame("ledger_summary")
        if df.empty or "n_executed" not in df.columns:
            return None, ""
        total = int(pd.to_numeric(df["n_executed"], errors="coerce").fillna(0).sum())
        if "purpose" in df.columns:
            by = (
                df.assign(_n=pd.to_numeric(df["n_executed"], errors="coerce").fillna(0))
                .groupby(df["purpose"].astype(str))["_n"]
                .sum()
            )
            parts = ", ".join(f"{p} {int(v):,}" for p, v in by.items() if v)
        else:
            parts = ""
        return total, parts


def _finish(
    ctx: _Ctx,
    tid: str,
    df: pd.DataFrame,
    extended: pd.DataFrame | None,
    notes: list[str],
    ext_notes: list[str] | None = None,
    caption: str = "",
    reported: bool = False,
    title: str | None = None,
) -> TableSpec:
    """Attach the "—" note and the provenance trailer (‡ / synthetic notes) to both versions; a bundle analysed
    with stale scores gets the ``STALE SCORES`` note first."""
    dash_note = (
        "— = not available in this run (no data for that cell, e.g. no pairs at that age or no accepts)."
    )

    def assemble(frames: Sequence[pd.DataFrame | None], extra: list[str]) -> list[str]:
        out = ctx.stale_note(reported) + list(notes) + list(extra)
        if not reported:
            if any(_contains(f, DASH) for f in frames):
                out.append(dash_note)
            out += ctx.validity_notes()
        return out + ctx.trailer(any(_contains(f, DAGGER) for f in frames), reported=reported)

    foot = assemble([df], [])
    efoot = assemble([extended], list(ext_notes or [])) if extended is not None else None
    return TableSpec(
        id=tid,
        title=title or TITLES.get(tid, tid),
        df=df,
        extended=extended,
        footnotes=foot,
        caption=caption,
        extended_footnotes=efoot,
        reported=reported,
        synthetic=ctx.synthetic,
    )


# --------------------------------------------------------------------------- T1 / T2


def _decoding_desc(ctx: _Ctx, env_id: str) -> str:
    e = ctx.envs.get(env_id)
    if e is None:
        return DASH
    t = e["temperature"]
    head = "Greedy (temperature 0, " if math.isfinite(t) and t == 0 else f"Temperature {t:g} sampling ("
    top_k = "off" if _num(e["top_k"]) == 0 else fmt_int(e["top_k"])
    mx = fmt_int(e["max_new_tokens"])
    return (
        f"{head}top-p {fmt_num(e['top_p'], 1)}, top-k {top_k}, repetition penalty "
        f"{fmt_num(e['repetition_penalty'], 1)}); max {mx} new tokens"
    )


def _versions(ctx: _Ctx) -> str:
    pk = ctx.prov.get("packages") or {}
    parts = [f"driftlab {ctx.meta.get('driftlab_version', ctx.prov.get('driftlab_version', '?'))}"]
    py = _dig(ctx.prov, "python", "version")
    if py:
        parts.append(f"Python {py}")
    for name in ("numpy", "pandas", "pydantic", "torch", "transformers", "vllm"):
        if pk.get(name):
            parts.append(f"{name} {pk[name]}")
    kind = _dig(ctx.prov, "backend", "kind") or _dig(ctx.cfg, "backend", "kind")
    if kind:
        parts.append(f"backend {kind}")
    return "; ".join(parts)


def _gpu(ctx: _Ctx) -> str:
    smi = _dig(ctx.prov, "gpu", "nvidia_smi")
    if isinstance(smi, list) and smi:
        return "; ".join(_text(g.get("name"), "?") for g in smi if isinstance(g, Mapping))
    dev = _dig(ctx.prov, "gpu", "torch_cuda", "device_name")
    return str(dev) if dev else "none detected (CPU)"


def _short_ts(ts: Any) -> str:
    """``"2026-10-05T12:00:00+00:00"`` -> ``"2026-10-05T12:00Z"`` (unparsable: the text itself, or ``"?"``)."""
    from datetime import datetime, timezone

    s = _text(ts).strip()
    if not s:
        return "?"
    try:
        dt = datetime.fromisoformat(s[:-1] + "+00:00" if s.endswith(("Z", "z")) else s)
    except ValueError:
        return s
    dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%MZ")


def _repo_relative(path: str) -> str:
    """``path`` relative to the repository root when it lies inside it (no local prefixes in tables)."""
    if not path:
        return ""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except (ValueError, OSError):
        return path


def _sha8(v: Any) -> str:
    s = _text(v)
    return f"{s[:8]}…" if s else "?"


def prereg_text(meta: Mapping[str, Any]) -> str:
    """Human-readable pre-registration status (T1 row ``Pre-registration``) from ``meta["plan_lock"]``."""
    lock = meta.get("plan_lock") if isinstance(meta, Mapping) else None
    if not isinstance(lock, Mapping) or not lock.get("status"):
        return "Not checked: the bundle was written before the pre-registration lock check"
    version = _text(meta.get("plan_version"), "?") or "?"
    status = _text(lock.get("status"))
    sha = _sha8(lock.get("plan_sha256"))
    at = _short_ts(lock.get("locked_at"))
    if status == "locked_before_run":
        return f"Plan {version} locked {at} before the run started (sha256 {sha})"
    if status == "locked_after_run_start":
        return (
            f"Plan {version} locked {at}, after the run started ({_short_ts(lock.get('run_created_at'))}): "
            f"not a pre-registration of this run (sha256 {sha})"
        )
    if status == "no_lock":
        return "No lock file: analysis plan not frozen"
    if status == "hash_mismatch":
        name = Path(_text(lock.get("lock_path"))).name or "the lock file"
        return (
            f"Lock file {name} freezes a different plan (sha256 {_sha8(lock.get('lock_sha256'))}) than the one "
            f"analysed (sha256 {sha}): analysis plan not frozen"
        )
    return f"Unknown lock status {status!r}: analysis plan not verified as frozen"


def _build_t1(ctx: _Ctx) -> TableSpec:
    cfg, plan = ctx.cfg, ctx.plan
    model = cfg.get("model") or {}
    data = cfg.get("data") or {}
    st = str(plan.storage_env)
    st_dec = (ctx.envs.get(st) or {}).get("decoding")
    alt = []
    for eid, e in ctx.envs.items():
        if e["decoding"] != st_dec and e["decoding"] not in [ctx.envs[a]["decoding"] for a in alt]:
            alt.append(eid)
    extractors: dict[str, str] = {}
    for e in ctx.envs.values():
        extractors.setdefault(e["extractor"], e["tag"])
    ext_text = "; ".join(
        f"{x} {EXTRACTOR_WORDS.get(x, '')} ({tag})".replace("  ", " ") for x, tag in extractors.items()
    )
    traj = cfg.get("trajectory") or {}
    prop = traj.get("proposer") or {}
    if traj.get("mode") == "static":
        strategy = "Static: every round proposes a candidate from the initial prompt"
    else:
        strategy = (
            f"LLM meta-prompt revision of the incumbent from {prop.get('n_errors', '?')} sampled dev errors "
            f"(proposer T = {prop.get('temperature', '?')}, up to {prop.get('max_attempts', '?')} attempts, "
            f"seeded fallback edit otherwise); the candidate becomes the dev incumbent when "
            f"{_ADVANCE.get(str(traj.get('advance_rule')), str(traj.get('advance_rule')))}"
        )
    strategy += f"; {ctx.R} rounds; dry run (nothing is promoted automatically)"
    dev, ev = data.get("dev") or {}, data.get("eval") or {}
    seeds = ", ".join(str(s) for s in ctx.seeds)
    values = {
        "Target model": _text(model.get("id"), DASH),
        "Model checkpoint / revision": _text(model.get("revision"), DASH),
        "Dataset and configuration": (
            f"{data.get('dataset', '?')} ({data.get('config', '?')} configuration), revision "
            f"{_text(data.get('revision'))[:12] or '?'}"
        ),
        "Development examples": f"First {ctx.D} questions of the {dev.get('split', 'train')} split "
        "(prompt optimization only)",
        "Evaluation examples": f"First {ctx.N} questions of the {ev.get('split', 'test')} split "
        "(held out; measurement only)",
        "Baseline decoding": _decoding_desc(ctx, st),
        "Alternative decoding": "; ".join(_decoding_desc(ctx, a) for a in alt) or DASH,
        "Answer extraction versions": ext_text or DASH,
        "Number of random seeds": f"{ctx.S} (seeds {seeds})" if ctx.S else DASH,
        "Prompt revision strategy": strategy,
        "Software / package versions": _versions(ctx),
        "Pre-registration": prereg_text(ctx.meta),
    }
    rows = [[k, values[k]] for k in T1_ROWS]
    matrix = cfg.get("matrix") or {}
    git = ctx.prov.get("git") or {}
    commit = _text(git.get("commit"), "unknown")
    if git.get("dirty"):
        commit += " (uncommitted changes)"
    extra = [
        ["Rounds per seed (R)", str(ctx.R)],
        ["Max new tokens", fmt_int(model.get("max_new_tokens"))],
        [
            "Trajectory mode / advance rule",
            f"{traj.get('mode', '?')} / {traj.get('advance_rule', '?')} (environment {traj.get('env', '?')})",
        ],
        ["Promotion rule", f"{plan.promotion_rule.kind}: {ctx.rule_text()}"],
        ["Ground-truth mode", plan.gt.mode],
        [
            "Canonical environment (final accuracy)",
            f"{plan.gt.canonical_env} ({ctx.env_short(plan.gt.canonical_env)})",
        ],
        ["Storage environment (references)", f"{st} ({ctx.env_short(st)})"],
        ["Environment schedule (policy simulation)", ctx.schedule_text()],
        ["Policies simulated", ", ".join(str(p) for p in ctx.meta.get("policies") or plan.policies)],
        ["Cost convention", plan.cost_convention],
        [
            "Measurement matrix",
            f"{matrix.get('mode', '?')}; physical greedy reruns {matrix.get('physical_greedy_reruns', '?')} "
            f"{matrix.get('physical_ages', '')}; GT draws {matrix.get('gt_draws', '?')}",
        ],
        [
            "Bootstrap replicates (B)",
            f"{ctx.B:,} (seed {ctx.meta.get('bootstrap_seed', plan.bootstrap.seed)})",
        ],
        [
            "Analysis plan hash",
            f"{ctx.bundle.plan_hash or '?'} ({ctx.meta.get('plan_version', plan.version)})",
        ],
        ["Plan lock file", _repo_relative(_text((ctx.plan_lock or {}).get("lock_path"))) or DASH],
        ["Config hash", str(ctx.meta.get("config_hash", "?"))],
        ["Git commit", commit],
        ["Engine fingerprint", _text(ctx.meta.get("engine_fp"), "?")],
        ["GPU", _gpu(ctx)],
        ["Run id", ctx.run_id],
        ["Synthetic data", "Yes (mock backend)" if ctx.synthetic else "No"],
        ["Analysis warnings", _warnings_text(ctx.meta.get("warnings"))],
    ]
    df = _frame(rows, PAPER_COLUMNS["T1"])
    ext = _frame(rows + extra, PAPER_COLUMNS["T1"])
    notes = [
        "Development questions are used only to build the prompt trajectory; every reported metric uses the "
        "evaluation questions.",
        "Every decoding parameter is passed explicitly (the checkpoint's generation_config defaults are never "
        "used); the answer extractors are frozen and identified by a source hash (name@hash).",
    ]
    return _finish(
        ctx,
        "T1",
        df,
        ext,
        notes,
        caption="Fixed configuration of the dry-run prompt-level self-improvement loop.",
    )


def _warnings_text(warnings: Any) -> str:
    """``"none"`` or ``"2: first warning; second warning"`` from ``meta["warnings"]``."""
    items = [str(w) for w in warnings] if isinstance(warnings, (list, tuple)) else []
    return f"{len(items)}: " + "; ".join(items) if items else "none"


_ADVANCE = {
    "dev_gt": "its dev accuracy exceeds the incumbent's",
    "dev_ge": "its dev accuracy is at least the incumbent's",
    "always": "it is proposed (always advance)",
}


def _build_t2(ctx: _Ctx) -> TableSpec:
    st = str(ctx.plan.storage_env)
    ref_version = f"{st}: {ctx.env_short(st)}, stored, scores kept" if st in ctx.envs else DASH
    rows, ext_rows = [], []
    for eid in ctx.envs:
        e = ctx.envs[eid]
        cur = ctx.env_short(eid)
        rows.append([eid, ctx.env_label(eid), ref_version, cur, ctx.env_purpose(eid)])
        ext_rows.append(
            [
                eid,
                ctx.env_label(eid),
                e["decoding"] or DASH,
                fmt_num(e["temperature"], 1),
                fmt_num(e["top_p"], 2),
                fmt_int(e["top_k"]),
                fmt_num(e["repetition_penalty"], 2),
                fmt_int(e["max_new_tokens"]),
                e["tag"] or DASH,
                ", ".join(e["changed"]) or "none",
                e["fingerprint"] or DASH,
                ref_version,
                cur,
                ctx.env_purpose(eid),
            ]
        )
    ext_cols = (
        "Condition ID",
        "Environment condition",
        "Decoding",
        "Temperature",
        "Top-p",
        "Top-k",
        "Repetition penalty",
        "Max new tokens",
        "Extractor",
        "Changed vs storage",
        "Environment fingerprint",
        "Reference version",
        "Current environment",
        "Purpose",
    )
    notes = [
        f"Reference version: the stored reference was generated under {st} and keeps the scores it had at "
        "storage time; it is compared with candidates evaluated under the current environment.",
        "Extractor tags are name@source-hash: v1 strict and v2 lenient are frozen (v1-correct implies v2-correct).",
        f"Environment schedule of the policy simulation: {ctx.schedule_text()}.",
    ]
    ext_notes = [
        "Fingerprint = hash of (model, engine, decoding parameters, max new tokens, extractor@hash); a policy "
        "detects an environment change when the fingerprint differs from the reference's."
    ]
    return _finish(
        ctx,
        "T2",
        _frame(rows, PAPER_COLUMNS["T2"]),
        _frame(ext_rows, ext_cols),
        notes,
        ext_notes,
        caption="Predefined environment conditions (decoding × answer extractor), independent of candidate "
        "performance.",
    )


# --------------------------------------------------------------------------- T3 / T6 (policies)


def _coupled_all(row: Mapping[str, Any] | None) -> bool:
    """Every accept of the policy is coupled to the GT comparison (so a zero FAR is by construction)."""
    if row is None:
        return False
    n_acc = _int(row.get("n_accepted"))
    return n_acc > 0 and _int(row.get("n_accepted_gt_coupled"), -1) == n_acc


def _build_t3(ctx: _Ctx) -> TableSpec:
    summ = ctx.summary_rows()
    rows = []
    for p in ctx.headline:
        r = summ.get(p)
        far = mean_sd_str(_get(r, "far_seed_mean"), _get(r, "far_seed_sd"), 1)
        rows.append(
            [
                f"{p}: {ctx.policy_label(p, r)}",
                ctx.policy_rule(p, r),
                mean_sd_str(_get(r, "gt_acc_mean"), _get(r, "gt_acc_sd"), 1),
                mark(far, _coupled_all(r)),
                fmt_int(_get(r, "calls_total")),
                fmt_count(_get(r, "refreshes")),
            ]
        )
    executed, breakdown = ctx.ledger_executed()
    ext_rows = []
    order = list(summ) or ctx.headline
    for p in order:
        r = summ.get(p)
        far = mean_sd_str(_get(r, "far_seed_mean"), _get(r, "far_seed_sd"), 1)
        pooled = with_ci(_get(r, "far_pooled"), _get(r, "far_lo"), _get(r, "far_hi"), 1)
        coupled = _coupled_all(r)
        ext_rows.append(
            [
                f"{p}: {ctx.policy_label(p, r)}",
                ctx.policy_rule(p, r),
                yes_no(_get(r, "adopt_on_promote", None)),
                mean_sd_str(_get(r, "gt_acc_mean"), _get(r, "gt_acc_sd"), 1),
                mean_sd_str(_get(r, "gt_acc_final_mean"), _get(r, "gt_acc_final_sd"), 1),
                mark(far, coupled),
                f"{fmt_int(_get(r, 'n_seeds_far'))}/{fmt_int(_get(r, 'n_seeds'))}" if r else DASH,
                frac_str(_get(r, "n_false_accepts"), _get(r, "n_accepted"), True),
                mark(pooled, coupled),
                frac_str(_get(r, "n_accepted_gt_coupled"), _get(r, "n_accepted"), True),
                fmt_count(_get(r, "calls_candidate_generation"), True),
                fmt_count(_get(r, "calls_evaluation"), True),
                fmt_count(_get(r, "calls_reference"), True),
                fmt_count(_get(r, "calls_total"), True),
                fmt_count(_get(r, "refreshes"), True),
                fmt_count(_get(r, "rescores"), True),
                fmt_int(executed, True) if executed is not None else DASH,
            ]
        )
    ext_cols = (
        "Policy",
        "Reference refresh rule",
        "Adopts candidate outputs on promotion",
        "GT accuracy, canonical env (%)",
        "GT accuracy, final-round env (%)",
        "False-accept rate, mean ± sd over seeds (%)",
        "Seeds with ≥ 1 accept",
        "False accepts / accepts (pooled)",
        "False-accept rate, pooled (%) [95% Wilson CI]",
        "Accepts coupled to GT",
        "Candidate-generation calls",
        "Evaluation calls",
        "Reference calls",
        "Total model calls",
        "Reference refreshes",
        "Rescores (0 calls)",
        "Executed calls (whole run, shared)",
    )
    canon = ctx.plan.gt.canonical_env
    notes = [
        f"Mean ± standard deviation over {ctx.S} seeds ({ctx.N} test questions, {ctx.R_done} rounds each). "
        "False-accept rate is worked out per seed and then averaged (seeds without any accepted candidate are "
        "left out of the average). The frozen reference is the stored output of the round-0 prompt, so it also "
        "accepts candidates that beat the starting prompt but not the current one.",
        f"Ground-truth accuracy: final shadow incumbent on the test questions under the canonical environment "
        f"{canon} ({ctx.env_short(canon)}). A false accept is an accepted candidate whose ground-truth accuracy "
        "is no better than the incumbent's under the same environment.",
        ctx.convention_text(),
        f"Environment schedule: {ctx.schedule_text()}. Promotion rule: {ctx.rule_text()} (dry run: nothing is "
        "promoted for real).",
        *ctx.far_notes(ctx.headline),
    ]
    ext_notes = [
        *ctx.far_notes([p for p in order if p not in ctx.headline]),
        "Pooled false-accept rate = Σ false accepts / Σ accepts over seeds, with a Wilson 95% CI. Accepts coupled "
        "to GT: the decision reused the incumbent's own ground-truth generations (greedy, same items), so they "
        "cannot be false accepts; a zero FAR made only of such accepts is marked ‡.",
        "Candidate-generation calls = incumbent dev + proposer (+ candidate dev); evaluation = candidate "
        "evaluations (+ Oracle ground truth); reference = initial reference + refreshes.",
        (
            f"Executed calls: generations actually paid by this run (ledger cache misses) = {executed:,}"
            + (f" ({breakdown})" if breakdown else "")
            + "; every policy is a dry-run simulation over the same stored generations, so its model calls "
            "are a counterfactual logical cost."
        )
        if executed is not None
        else "Executed calls: no ledger in this bundle.",
    ]
    return _finish(
        ctx,
        "T3",
        _frame(rows, PAPER_COLUMNS["T3"]),
        _frame(ext_rows, ext_cols),
        notes,
        ext_notes,
        caption="Reference-refresh policies replayed on the same stored generations: final ground-truth "
        "accuracy, false-accept rate and model-call cost.",
    )


def _get(row: Mapping[str, Any] | None, key: str, default: Any = float("nan")) -> Any:
    if row is None:
        return default
    v = row.get(key, default)
    return default if v is None else v


def _build_t6(ctx: _Ctx) -> TableSpec:
    summ = ctx.summary_rows()

    def row_for(p: str, extended: bool) -> list[str]:
        r = summ.get(p)
        coupled = _coupled_all(r)
        n_acc = _int(_get(r, "n_accepted"), 0)
        far = fmt_pct(_get(r, "far_pooled")) if n_acc else DASH
        base = [
            fmt_int(_get(r, "n_evaluated"), extended),
            fmt_int(_get(r, "n_accepted"), extended),
            fmt_int(_get(r, "n_false_accepts"), extended),
        ]
        if not extended:
            return [ctx.policy_label(p, r), *base, mark(far, coupled)]
        pooled = with_ci(_get(r, "far_pooled"), _get(r, "far_lo"), _get(r, "far_hi"), 1) if n_acc else DASH
        return [
            p,
            ctx.policy_label(p, r),
            *base,
            mark(pooled, coupled),
            mark(mean_sd_str(_get(r, "far_seed_mean"), _get(r, "far_seed_sd"), 1), coupled),
            frac_str(_get(r, "n_accepted_gt_coupled"), _get(r, "n_accepted"), True),
        ]

    rows = [row_for(p, False) for p in ctx.headline]
    ext_rows = [row_for(p, True) for p in (list(summ) or ctx.headline)]
    ext_cols = (
        "Policy ID",
        "Refresh policy",
        "Candidates evaluated",
        "Candidates accepted",
        "Accepted candidates with no ground-truth improvement",
        "False-accept rate, pooled (%) [95% Wilson CI]",
        "False-accept rate, mean ± sd over seeds (%)",
        "Accepts coupled to GT",
    )
    notes = [
        f"Pooled over {ctx.S} seeds × {ctx.R_done} candidate rounds. False-accept rate = accepted candidates "
        "with no ground-truth improvement / candidates accepted; no ground-truth improvement means the "
        "candidate's ground-truth accuracy is at most the incumbent's under the round's environment.",
        "The frozen reference never changes, so it accepts candidates that only beat the round-0 prompt.",
        *ctx.far_notes(ctx.headline, seed_means=False),
    ]
    all_policies = list(summ) or ctx.headline
    ext_notes = [
        *ctx.far_notes([p for p in all_policies if p not in ctx.headline], seed_means=False),
        *(n for n in ctx.far_notes(all_policies) if n.startswith("False-accept rate averaged")),
    ]
    return _finish(
        ctx,
        "T6",
        _frame(rows, PAPER_COLUMNS["T6"]),
        _frame(ext_rows, ext_cols),
        notes,
        ext_notes,
        caption="Accepted candidates that did not improve ground-truth accuracy, per refresh policy.",
    )


# --------------------------------------------------------------------------- T4 / T7 / T7b (all pairs)


def _summary_frame(ctx: _Ctx) -> pd.DataFrame:
    df = ctx.frame("pair_summary")
    if df.empty or not {"env", "age"} <= set(df.columns):
        return pd.DataFrame(columns=["env", "age"])
    return df


def _lookup(df: pd.DataFrame, env: str, age: int) -> dict[str, Any] | None:
    if df.empty or not {"env", "age"} <= set(df.columns):
        return None
    m = (df["env"].astype(str) == str(env)) & (pd.to_numeric(df["age"], errors="coerce") == int(age))
    hit = df.loc[m]
    return None if hit.empty else hit.iloc[0].to_dict()


def _table4(ctx: _Ctx) -> pd.DataFrame:
    df = ctx.frame("table4")
    if not df.empty and "env" in df.columns:
        return df
    summ = _summary_frame(ctx)
    if summ.empty:
        return pd.DataFrame()
    from driftlab.analysis.allpairs import table4_frame

    return table4_frame(summ, ctx.plan, env_ids=ctx.env_ids())


def _table7(ctx: _Ctx) -> pd.DataFrame:
    df = ctx.frame("table7")
    if not df.empty and {"age", "env"} <= set(df.columns):
        return df
    from driftlab.analysis.allpairs import table7_frame

    return table7_frame(_summary_frame(ctx), ctx.plan)


def _pairs(ctx: _Ctx) -> pd.DataFrame:
    df = ctx.frame("pairs")
    need = {"env", "age", "seed", "inflation"}
    return df if not df.empty and need <= set(df.columns) else pd.DataFrame()


def _coupled_accepts(pairs: pd.DataFrame, env: str, age: int, ref: str) -> tuple[int, int] | None:
    """``(k, n)``: accepts against ``ref`` that cannot be false accepts by construction, and all accepts, of one
    (env, age) cell pooled over seeds (``None`` when the pairs frame lacks the needed columns).

    * fresh: GT is read from the decision cells (``gt_coupled``);
    * rerun: additionally the reference prompt is the current incumbent (rerun cell == fresh cell);
    * stored: additionally the stored generation is the rerun generation under the same extractor.
    """
    if pairs.empty or f"dec_{ref}" not in pairs.columns or "gt_coupled" not in pairs.columns:
        return None
    sub = pairs.loc[(pairs["env"].astype(str) == env) & (pd.to_numeric(pairs["age"]) == age)]
    acc = sub.loc[sub[f"dec_{ref}"].map(_truthy).astype(bool)]
    ok = acc["gt_coupled"].map(_truthy).astype(bool)
    if ref in ("rerun", "stored") and {"ref_slot", "cur_slot"} <= set(acc.columns):
        ok &= pd.to_numeric(acc["ref_slot"]) == pd.to_numeric(acc["cur_slot"])
    if ref == "stored" and "by_construction" in acc.columns:
        ok &= acc["by_construction"].map(_truthy).astype(bool)
    return int(ok.sum()), int(len(acc))


def _far_partial_notes(cells: Sequence[tuple[str, tuple[int, int] | None]]) -> list[str]:
    """Footnote for FAR cells where only SOME accepts cannot be false by construction (‡ needs every accept).

    ``cells``: ``(label, _coupled_accepts(...))``.
    """
    part = [f"{label} {c[0]} of {c[1]}" for label, c in cells if c is not None and 0 < c[0] < c[1]]
    if not part:
        return []
    return [
        "Partly zero by construction: some accepts behind these false-accept rates were decided against the "
        "current incumbent's own ground-truth generations (greedy decoding: the reference outputs are the current "
        "incumbent's generations of the candidate's round under the current extractor), so they cannot be false "
        "accepts and lower the rate by construction (‡ marks a rate made only of such accepts). Such accepts / "
        "all accepts, pooled over seeds: " + "; ".join(part) + "."
    ]


def _far_coupled(
    ctx: _Ctx, pairs: pd.DataFrame, env: str, age: int, ref: str, row: Mapping[str, Any]
) -> bool:
    """True when every pair accepted against ``ref`` could not be a false accept by construction
    (see :func:`_coupled_accepts`)."""
    counts = _coupled_accepts(pairs, env, age, ref)
    if counts is not None:
        k, n = counts
        return n > 0 and k == n
    # No pairs frame: only the structural cases are certain.
    if not _truthy(row.get("gt_coupled")):
        return False
    if ref == "fresh":
        return True
    return (
        age == 0
        and ctx.plan.reference_mode == "incumbent"
        and (ref == "rerun" or _truthy(row.get("by_construction")))
    )


def _far_ref(
    ctx: _Ctx,
    pairs: pd.DataFrame,
    row: Mapping[str, Any],
    s_row: Mapping[str, Any],
    env: str,
    age: int,
    ref: str,
) -> str:
    """FAR cell of one reference kind for a T7 row (counts from the pair summary when available)."""
    n, k = s_row.get(f"n_accept_{ref}"), s_row.get(f"n_fa_cur_{ref}")
    rate = row.get(f"far_cur_{ref}", s_row.get(f"far_cur_{ref}"))
    coupled = _far_coupled(ctx, pairs, env, age, ref, row)
    if not _finite(n):
        return mark(fmt_pct(rate), coupled)
    return _far_cell(rate, k, n, coupled, thousands=True)


def _share(v: Any) -> float:
    f = _num(v)
    return f if math.isfinite(f) else 0.0


def _construction_notes(cells: Sequence[tuple[str, Any, Any]]) -> list[str]:
    """Footnotes for cells whose pairs are only PARTLY identical by construction (‡ needs every pair).

    ``cells``: ``(label, by_construction_frac, gen_by_construction_frac)``. A pair is identical by construction
    when its rerun is the stored generation itself (cache hit) under the same extractor: it adds exactly 0 to the
    pooled inflation. A pair whose rerun is the stored generation under ANOTHER extractor has a generation part
    of 0 by construction, so it measures the extraction part only.
    """
    part, gen = [], []
    for label, bc, g in cells:
        bc, g = _share(bc), _share(g)
        if 0.0 < bc < 1.0:
            part.append(f"{label} {fmt_pct(bc, 0)}%")
        if g > bc + 1e-9:
            gen.append(f"{label} {fmt_pct(g - bc, 0)}%")
    out = []
    if part:
        out.append(
            "Partly identical by construction (‡ needs every pair): share of pairs whose rerun reference is the "
            "stored generation itself (cache hit, same extractor), which adds exactly 0 to the inflation: "
            + "; ".join(part)
            + "."
        )
    if gen:
        out.append(
            "Generation part zero by construction: share of pairs whose rerun reference is the stored generation "
            "re-scored by the current extractor (cache hit), so their inflation is the extraction part only: "
            + "; ".join(gen)
            + "."
        )
    return out


def _far_cell(rate: Any, k: Any, n: Any, coupled: bool, digits: int = 1, thousands: bool = False) -> str:
    """``"42.9 (3/7)"``; ``"—"`` without accepts."""
    if _int(n) <= 0 or not _finite(rate):
        return DASH
    return mark(f"{fmt_pct(rate, digits)} ({frac_str(k, n, thousands)})", coupled)


def _gen_frac(summ: pd.DataFrame, env: str, age: int, all_flag: Any) -> float:
    """Share of pairs whose generation part is 0 by construction (pair summary; else the all-pairs flag)."""
    s_row = _lookup(summ, env, age)
    if s_row is not None and _finite(s_row.get("gen_by_construction_frac")):
        return _num(s_row.get("gen_by_construction_frac"))
    return 1.0 if _truthy(all_flag) else 0.0


def _build_t4(ctx: _Ctx) -> TableSpec:
    t4 = _table4(ctx)
    summ = _summary_frame(ctx)
    st = str(ctx.plan.storage_env)
    st_ext = (ctx.envs.get(st) or {}).get("extractor")
    age = int(ctx.plan.t4.age)
    rows, ext_rows, cells = [], [], []
    for r in t4.to_dict("records") if not t4.empty else []:
        env = _text(r.get("env"))
        label = _text(r.get("env_label")) or env
        n_pairs = _int(r.get("n_pairs"))
        bc = _truthy(r.get("by_construction"))
        a = _int(r.get("age"), age)
        if n_pairs <= 0:
            rows.append([label, str(a), DASH, DASH, DASH])
            ext_rows.append([label, env, str(a), *([DASH] * 9), "0", fmt_int(r.get("n_seeds"))])
            continue
        cells.append(
            (label, r.get("by_construction_frac"), _gen_frac(summ, env, a, r.get("gen_by_construction")))
        )
        stored = mean_sd_str(r.get("stored_win_seed_mean"), r.get("stored_win_seed_sd"), 2)
        rerun = mark(mean_sd_str(r.get("rerun_win_seed_mean"), r.get("rerun_win_seed_sd"), 2), bc)
        infl = mark(mean_sd_str(r.get("infl_seed_mean"), r.get("infl_seed_sd"), 2), bc)
        rows.append([label, str(a), stored, rerun, infl])
        same_x = st_ext is not None and (ctx.envs.get(env) or {}).get("extractor") == st_ext
        pooled = "0.00" if bc else with_ci(r.get("inflation"), r.get("infl_lo"), r.get("infl_hi"), 2)
        ext_rows.append(
            [
                label,
                env,
                str(a),
                stored,
                rerun,
                infl,
                mark(pooled, bc),
                mark(fmt_num(r.get("infl_extract"), 2, 100), same_x),
                mark(fmt_num(r.get("infl_generation"), 2, 100), _truthy(r.get("gen_by_construction"))),
                fmt_pct(r.get("flip_rate")),
                f"{fmt_pct(r.get('stored_win'), 2)} / {fmt_pct(r.get('rerun_win'), 2)}",
                fmt_pct(r.get("by_construction_frac")),
                fmt_int(n_pairs, True),
                fmt_int(r.get("n_seeds")),
            ]
        )
    if not rows:
        from driftlab.analysis.allpairs import T4_LABELS

        rows = [[T4_LABELS.get(e, e), str(age), DASH, DASH, DASH] for e in ctx.env_ids()]
    ext_cols = (
        "Environment condition",
        "Env",
        "Reference age (rounds)",
        "Stored-reference win rate (%)",
        "Rerun-reference win rate (%)",
        "Drift inflation (pp)",
        "Pooled drift inflation (pp) [95% CI]",
        "Extraction part (pp)",
        "Generation part (pp)",
        "Decision flip rate (%)",
        "Pooled stored / rerun win rate (%)",
        "Pairs identical by construction (%)",
        "Pairs",
        "Seeds",
    )
    notes = [
        "Drift inflation = stored-reference win rate − rerun-reference win rate.",
        f"Each cell averages every candidate and reference pair that are {age} rounds apart. The reference "
        f"answers were stored under {st} and kept the scores they had at the time.",
        f"Mean ± standard deviation over {ctx.S} seeds of the per-seed means; win rate = share of the "
        f"{ctx.n_decision} test questions the candidate answers correctly and the reference does not. Rerun "
        "reference = the same reference prompt regenerated in the candidate's round under the current environment.",
        ctx.seed_sd_text(),
        *_construction_notes(cells),
        *ctx.skipped_note(),
    ]
    ext_notes = [
        f"Pooled inflation = Σ over pairs / Σ n; brackets: {ctx.bootstrap_text()}. Extraction part = stored − "
        "re-scored win rate (same text, current extractor); generation part = re-scored − rerun win rate; they sum "
        "to the inflation. Flip rate = share of pairs where the stored and the rerun reference lead to different "
        "promotion decisions."
    ]
    return _finish(
        ctx,
        "T4",
        _frame(rows, PAPER_COLUMNS["T4"]),
        _frame(ext_rows, ext_cols) if ext_rows else None,
        notes,
        ext_notes,
        caption=f"How much stored references inflate measured win rates after {age} rounds, per environment change.",
    )


def _seed_infl(summ: pd.DataFrame, env: str, age: int) -> str:
    r = _lookup(summ, env, age)
    if r is None:
        return DASH
    return mean_sd_str(r.get("infl_seed_mean"), r.get("infl_seed_sd"), 2)


def _build_t7(ctx: _Ctx) -> TableSpec:
    t7 = _table7(ctx)
    summ = _summary_frame(ctx)
    pairs = _pairs(ctx)
    plan = ctx.plan
    st, changed = str(plan.storage_env), str(plan.t7.changed_env)
    rows, ext_rows, cells = [], [], []
    far_paper: list[tuple[str, tuple[int, int] | None]] = []  # stored-reference FAR (paper + extended)
    far_ext: list[tuple[str, tuple[int, int] | None]] = []  # rerun / fresh FAR (extended only)
    for r in t7.to_dict("records") if not t7.empty else []:
        age = _int(r.get("age"))
        env = _text(r.get("env"))
        status = _text(r.get("env_status")) or env
        n_pairs = _int(r.get("n_pairs"))
        if n_pairs <= 0:
            rows.append([str(age), status, DASH, DASH, DASH, DASH])
            ext_rows.append([str(age), status, env, "0", "0", *([DASH] * 8), "0", DASH])
            continue
        bc = _truthy(r.get("by_construction"))
        cells.append((f"age {age}, {status}", r.get("by_construction_frac"), _gen_frac(summ, env, age, bc)))
        s_row = _lookup(summ, env, age) or {}
        n_acc = _int(r.get("n_accept_stored"))
        k_fa = s_row.get("n_fa_cur_stored")
        if not _finite(k_fa):
            k_fa = _num(r.get("far_cur_stored")) * n_acc if n_acc else float("nan")
        c_stored = _far_coupled(ctx, pairs, env, age, "stored", r)
        far_paper.append((f"age {age}, {status}", _coupled_accepts(pairs, env, age, "stored")))
        for ref in ("rerun", "fresh"):
            far_ext.append((f"age {age}, {status}, {ref} reference", _coupled_accepts(pairs, env, age, ref)))
        infl = "0.00" if bc else with_ci(r.get("inflation"), r.get("infl_lo"), r.get("infl_hi"), 2)
        far = _far_cell(r.get("far_cur_stored"), k_fa, n_acc, c_stored)
        rows.append(
            [
                str(age),
                status,
                fmt_pct(r.get("stored_win"), 2),
                mark(fmt_pct(r.get("rerun_win"), 2), bc),
                mark(infl, bc),
                far,
            ]
        )

        stored_wilson = (
            mark(
                with_ci(r.get("far_cur_stored"), r.get("far_cur_stored_lo"), r.get("far_cur_stored_hi"), 1),
                c_stored,
            )
            if n_acc
            else DASH
        )
        ext_rows.append(
            [
                str(age),
                status,
                env,
                fmt_int(n_pairs, True),
                fmt_int(r.get("n_seeds")),
                fmt_pct(r.get("stored_win"), 2),
                mark(fmt_pct(r.get("rerun_win"), 2), bc),
                mark(infl, bc),
                mark(_seed_infl(summ, env, age), bc),
                stored_wilson,
                _far_ref(ctx, pairs, r, s_row, env, age, "rerun"),
                _far_ref(ctx, pairs, r, s_row, env, age, "fresh"),
                fmt_pct(r.get("flip_rate")),
                fmt_int(n_acc, True),
                fmt_pct(r.get("by_construction_frac")),
            ]
        )
    if not rows:
        for age in plan.t7.ages:
            for label in (f"Unchanged ({plan.t7.unchanged_env})", f"Changed ({changed})"):
                rows.append([str(age), label, DASH, DASH, DASH, DASH])
    ext_cols = (
        "Reference age (optimization rounds)",
        "Environment status",
        "Env",
        "Pairs",
        "Seeds",
        "Stored-reference win rate (%)",
        "Rerun-reference win rate (%)",
        "Drift inflation (pp) [95% CI]",
        "Drift inflation, mean ± sd over seeds (pp)",
        "False-accept rate, stored reference (%) [95% Wilson CI]",
        "False-accept rate, rerun reference (%) (k/n)",
        "False-accept rate, fresh reference (%) (k/n)",
        "Decision flip rate (%)",
        "Accepts (stored reference)",
        "Pairs identical by construction (%)",
    )
    st_desc = ctx.env_short(st)
    whose = (
        "the incumbent's outputs"
        if plan.reference_mode == "incumbent"
        else "the outputs of the prompt proposed k rounds earlier (chain mode)"
    )
    notes = [
        f"Stored reference: {whose} generated k rounds before the candidate's round under the storage environment "
        f"{st} ({st_desc}), with the scores they had at storage time. Rerun reference: the same prompt regenerated "
        "in the candidate's round under the environment shown.",
        f"Drift inflation = stored-reference win rate − rerun-reference win rate, pooled over every candidate and "
        f"reference pair that are k rounds apart in all {ctx.S} seeds ({ctx.n_decision} test questions each); "
        f"brackets: {ctx.bootstrap_text()}.",
        ctx.seed_sd_text(),
        "False-accept rate: share of candidates accepted against the stored reference whose ground-truth accuracy "
        "is no better than the current incumbent's under the same environment (false accepts / accepts in "
        "parentheses, pooled over seeds).",
        f"Age 0: the reference is generated in the round it is used, so in a real loop no environment change can "
        f"occur between storage and evaluation. The Changed ({changed}) row at age 0 is the factorial "
        f"counterfactual (reference stored under {st}, compared under {changed} in the same round): it isolates "
        "the environment effect without any prompt staleness.",
        *_construction_notes(cells),
        *_far_partial_notes(far_paper),
        *ctx.skipped_note(),
    ]
    empty_ages = sorted(
        {
            _int(r.get("age"))
            for r in (t7.to_dict("records") if not t7.empty else [])
            if _int(r.get("n_pairs")) <= 0
        }
    )
    if empty_ages or t7.empty:
        listed = ", ".join(str(a) for a in empty_ages) or ", ".join(str(a) for a in plan.t7.ages)
        why = (
            f" (the analysed trajectories have {ctx.R_done} rounds, so no reference can be older than "
            f"{ctx.R_done} rounds)"
            if empty_ages and min(empty_ages) > ctx.R_done
            else ""
        )
        notes.append(f"No candidate/reference pair at age {listed}{why}: the rows are kept and shown as —.")
    ext_notes = [
        "Rerun reference FAR: decisions against the same prompt regenerated under the current environment; fresh "
        "reference FAR: decisions against the current incumbent regenerated under the current environment. "
        "Under greedy decoding the ground truth is read from the very generations the fresh reference uses, so "
        "its false accepts are impossible (‡). Flip rate = share of pairs where the stored and rerun references "
        "give different decisions.",
        *_far_partial_notes(far_ext),
    ]
    return _finish(
        ctx,
        "T7",
        _frame(rows, PAPER_COLUMNS["T7"]),
        _frame(ext_rows, ext_cols) if ext_rows else None,
        notes,
        ext_notes,
        caption=f"Win-rate inflation and false accepts as the stored reference ages, with the environment "
        f"unchanged ({plan.t7.unchanged_env}) or changed ({changed}).",
    )


def _build_t7b(ctx: _Ctx) -> TableSpec:
    summ = _summary_frame(ctx)
    pairs = _pairs(ctx)
    envs = ctx.env_ids()
    if not summ.empty:
        envs = list(dict.fromkeys([*envs, *summ["env"].astype(str)]))
    ages = sorted(
        {int(a) for a in ctx.plan.t7.ages}
        | (
            {int(a) for a in pd.to_numeric(summ["age"], errors="coerce").dropna()}
            if not summ.empty
            else set()
        )
    )
    from driftlab.analysis.allpairs import T4_LABELS

    rows, ext_rows, cells = [], [], []
    far_cells: list[tuple[str, tuple[int, int] | None]] = []
    for age in ages:
        for env in envs:
            label = f"{env} ({T4_LABELS.get(env, ctx.env_label(env))})"
            r = _lookup(summ, env, age)
            if r is None or _int(r.get("n_pairs")) <= 0:
                rows.append([str(age), label, *([DASH] * 7), "0", "0"])
                ext_rows.append([str(age), label, *([DASH] * 7), "0", "0", *([DASH] * 6)])
                continue
            bc = _truthy(r.get("all_by_construction"))
            cells.append(
                (f"{env} at age {age}", r.get("by_construction_frac"), r.get("gen_by_construction_frac"))
            )
            infl = "0.00" if bc else with_ci(r.get("inflation"), r.get("infl_lo"), r.get("infl_hi"), 2)
            fars = []
            for ref in ("stored", "rerun", "fresh"):
                coupled = _far_coupled(
                    ctx,
                    pairs,
                    env,
                    age,
                    ref,
                    {"gt_coupled": _num(r.get("gt_coupled_frac")) == 1.0, "by_construction": bc},
                )
                far_cells.append(
                    (f"{env} at age {age}, {ref} reference", _coupled_accepts(pairs, env, age, ref))
                )
                fars.append(
                    _far_cell(
                        r.get(f"far_cur_{ref}"), r.get(f"n_fa_cur_{ref}"), r.get(f"n_accept_{ref}"), coupled
                    )
                )
            base = [
                str(age),
                label,
                fmt_pct(r.get("win_stored"), 2),
                mark(fmt_pct(r.get("win_rerun"), 2), bc),
                mark(infl, bc),
                *fars,
                fmt_pct(r.get("flip_rate")),
            ]
            rows.append([*base, fmt_int(r.get("n_pairs")), fmt_int(r.get("n_accept_stored"))])
            same_x = (ctx.envs.get(env) or {}).get("extractor") == (
                ctx.envs.get(str(ctx.plan.storage_env)) or {}
            ).get("extractor")
            ext_rows.append(
                [
                    *base,
                    fmt_int(r.get("n_pairs"), True),
                    fmt_int(r.get("n_accept_stored"), True),
                    mark(fmt_num(r.get("infl_extract"), 2, 100), same_x and env in ctx.envs),
                    mark(
                        fmt_num(r.get("infl_generation"), 2, 100),
                        _num(r.get("gen_by_construction_frac")) == 1.0,
                    ),
                    mark(mean_sd_str(r.get("infl_seed_mean"), r.get("infl_seed_sd"), 2), bc),
                    ci_str(r.get("far_cur_stored_lo"), r.get("far_cur_stored_hi"), 1),
                    fmt_pct(r.get("by_construction_frac")),
                    fmt_pct(r.get("gt_coupled_frac")),
                ]
            )
    ext_cols = (
        *PAPER_COLUMNS["T7b"],
        "Extraction part (pp)",
        "Generation part (pp)",
        "Drift inflation, mean ± sd over seeds (pp)",
        "False-accept rate, stored: 95% Wilson CI (%)",
        "Pairs identical by construction (%)",
        "Pairs with GT coupled to the decision (%)",
    )
    notes = [
        f"Every reference age 0..{ctx.R_done} of the factorial all-pairs analysis (plus the Table 7 ages) × "
        f"environment; references stored under {ctx.plan.storage_env} with storage-time scores. Pooled over seeds; "
        f"inflation brackets: {ctx.bootstrap_text()}.",
        "False-accept rates are against the current incumbent's ground truth: stored reference (scores kept), "
        "rerun reference (same prompt regenerated now) and fresh reference (current incumbent regenerated now); "
        "false accepts / accepts in parentheses.",
        *_construction_notes(cells),
        *_far_partial_notes(far_cells),
        *ctx.skipped_note(),
    ]
    return _finish(
        ctx,
        "T7b",
        _frame(rows, PAPER_COLUMNS["T7b"]),
        _frame(ext_rows, ext_cols),
        notes,
        [ctx.seed_sd_text()],  # the extended grid adds a mean ± sd over seeds column
        caption="Full grid behind Table 7.",
    )


# --------------------------------------------------------------------------- T5


def _build_t5(ctx: _Ctx) -> TableSpec:
    df = ctx.frame("candidates")
    base_cols = {
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
    }
    dec_cols = [c for c in df.columns if c not in base_cols] if not df.empty else []
    preferred = [*ctx.headline, "ORACLE"]
    dec_cols = [c for c in preferred if c in dec_cols] + [c for c in dec_cols if c not in preferred]
    rows, ext_rows = [], []
    for r in df.to_dict("records") if not df.empty else []:
        cid = _text(r.get("candidate_id")) or f"s{_int(r.get('seed'))}-r{_int(r.get('round'))}"
        outcome = OUTCOME_LABELS.get(_text(r.get("gt_outcome")), _text(r.get("gt_outcome"), DASH) or DASH)
        delta = fmt_num(r.get("delta_pp"), 1, signed=True)
        rows.append(
            [
                cid,
                fmt_int(r.get("seed")),
                fmt_pct(r.get("incumbent_acc")),
                fmt_pct(r.get("candidate_acc")),
                delta,
                outcome,
            ]
        )
        decisions = [_text(r.get(c)).capitalize() or DASH for c in dec_cols]
        ext_rows.append(
            [
                cid,
                fmt_int(r.get("seed")),
                fmt_int(r.get("round")),
                _text(r.get("env"), DASH) or DASH,
                fmt_int(r.get("incumbent_slot")),
                fmt_pct(r.get("incumbent_acc")),
                fmt_pct(r.get("candidate_acc")),
                delta,
                outcome,
                fmt_num(r.get("dev_delta"), 1, 100, signed=True),
                yes_no(r.get("advanced")),
                *decisions,
            ]
        )
    ext_cols = (
        "Candidate ID",
        "Seed",
        "Round",
        "Environment at round",
        "Incumbent slot",
        "Incumbent accuracy (%)",
        "Candidate accuracy (%)",
        "Accuracy change (pp)",
        "Ground-truth outcome",
        "Dev accuracy change (pp)",
        "Advanced on dev",
        *(f"{c} decision" for c in dec_cols),
    )
    notes = [
        f"One row per candidate (seed s, round r: ID s<seed>-r<round>). Incumbent = the trajectory incumbent when "
        f"the candidate is evaluated; accuracies are ground-truth accuracies on the {ctx.N} test questions under "
        f"the environment of that round ({ctx.schedule_text()}; ground-truth mode {ctx.plan.gt.mode}).",
        "Ground-truth outcome: Improvement = candidate more accurate than the incumbent, No change = equal, "
        "Regression = less accurate.",
    ]
    ext_notes = [
        "Advanced on dev = the candidate replaced the trajectory incumbent on the development split (Phase A, "
        f"{_dig(ctx.cfg, 'trajectory', 'env', default='E1')}); decision columns are the dry-run decisions of each "
        "policy (Oracle = accept iff ground-truth accuracy improves)."
    ]
    return _finish(
        ctx,
        "T5",
        _frame(rows, PAPER_COLUMNS["T5"]),
        _frame(ext_rows, ext_cols) if ext_rows else None,
        notes,
        ext_notes,
        caption="Ground-truth accuracy of every proposed candidate against the incumbent it would replace.",
    )


# --------------------------------------------------------------------------- T8


def _fixed_age_ablation(r: Mapping[str, Any]) -> bool:
    """An A4-style ablation (its third policy is the fixed-age policy, no reference-age variation)."""
    if _text(r.get("p3_policy")).upper().startswith("FIXEDAGE"):
        return True
    v = r.get("reference_age_variation")
    return v is not None and _text(v) != "" and not _truthy(v)


def _ablation_construction(pairs: pd.DataFrame, r: Mapping[str, Any]) -> tuple[float, float]:
    """(by-construction share, generation-part-by-construction share) of the pairs behind an ablation's
    drift inflation (``inflation_age`` x ``inflation_envs``); NaN without a pairs frame or matching pairs."""
    envs = [e.strip() for e in _text(r.get("inflation_envs")).split(",") if e.strip()]
    if (
        pairs.empty
        or not envs
        or not _finite(r.get("inflation_age"))
        or "by_construction" not in pairs.columns
    ):
        return float("nan"), float("nan")
    m = pairs["env"].astype(str).isin(envs) & (pd.to_numeric(pairs["age"]) == _int(r.get("inflation_age")))
    sub = pairs.loc[m]
    if sub.empty:
        return float("nan"), float("nan")
    bc = float(sub["by_construction"].map(_truthy).mean())
    gen = (
        float(sub["gen_by_construction"].map(_truthy).mean()) if "gen_by_construction" in sub.columns else bc
    )
    return bc, gen


def _build_t8(ctx: _Ctx) -> TableSpec:
    df = ctx.frame("ablations")
    pairs = _pairs(ctx)
    rows, ext_rows, cells, part = [], [], [], []
    has_coupling = "n_accepted_gt_coupled_P1b" in df.columns and "n_accepted_gt_coupled_P3" in df.columns
    for r in df.to_dict("records") if not df.empty else []:
        name = _text(r.get("ablation"))
        label = ABLATION_LABELS.get(name) or _text(r.get("description")) or name
        pol = "P3" if _fixed_age_ablation(r) else "P1b"
        far = r.get(f"far_{pol}")
        n_acc, k = _int(r.get(f"n_accepted_{pol}")), _int(r.get(f"n_accepted_gt_coupled_{pol}"), -1)
        coupled = has_coupling and n_acc > 0 and k == n_acc
        if has_coupling and 0 < k < n_acc:
            part.append(f"{name} {k} of {n_acc}")
        bc, gen = _ablation_construction(pairs, r)
        cells.append((name, bc, gen))
        infl = mark(fmt_num(r.get("drift_inflation_pp"), 2), bc == 1.0)
        rows.append(
            [
                f"{name}: {label}",
                yes_no(r.get("decoding_changes")),
                yes_no(r.get("extraction_changes")),
                yes_no(r.get("reference_age_variation")),
                infl,
                mark(fmt_pct(far), coupled),
            ]
        )
        third = _text(r.get("p3_policy"), "P3") or "P3"
        ext_rows.append(
            [
                f"{name}: {label}",
                _text(r.get("schedule"), DASH) or DASH,
                yes_no(r.get("decoding_changes")),
                yes_no(r.get("extraction_changes")),
                yes_no(r.get("reference_age_variation")),
                fmt_int(r.get("inflation_age")),
                _text(r.get("inflation_envs")) or "none",
                fmt_int(r.get("n_pairs"), True),
                infl,
                fmt_pct(r.get("far_P1")),
                fmt_pct(r.get("far_P1b")),
                f"{fmt_pct(r.get('far_P3'))} ({third})",
                f"{fmt_int(r.get('n_accepted_P1'))} / {fmt_int(r.get('n_accepted_P1b'))} / "
                f"{fmt_int(r.get('n_accepted_P3'))}",
                fmt_pct(r.get("gt_acc_P3")),
                fmt_count(r.get("calls_P3"), True),
            ]
        )
    if not rows:
        rows = [[f"{a}: {lab}", DASH, DASH, DASH, DASH, DASH] for a, lab in ABLATION_LABELS.items()]
    ext_cols = (
        "Ablation",
        "Environment schedule",
        "Decoding changes enabled",
        "Extraction changes enabled",
        "Reference-age variation enabled",
        "Inflation age (rounds)",
        "Inflation environments",
        "Pairs",
        "Drift inflation (pp)",
        "False-accept rate, P1 frozen (%)",
        "False-accept rate, P1b frozen + adopt (%)",
        "False-accept rate, P3 / fixed-age policy (%)",
        "Accepts P1 / P1b / P3",
        "GT accuracy, P3 / fixed-age (%)",
        "Model calls, P3 / fixed-age",
    )
    fixed = [a.fixed_age for a in ctx.plan.ablations.values() if a.fixed_age is not None]
    notes = [
        f"Drift inflation: mean over candidate/reference pairs at reference age {ctx.plan.t4.age}"
        + (f" (fixed-age ablation: age {fixed[0]})" if fixed else "")
        + " in the environments the ablation's schedule visits from its first change on (— when the schedule "
        "changes nothing within the run).",
        "False-accept rate (pooled over seeds): frozen reference with adopt-on-promote (P1b), which isolates "
        "environment staleness from prompt staleness, for the variable-age ablations; for the fixed-age ablation, "
        "the fixed-age policy (reference = incumbent output stored that many rounds earlier).",
        f"Enabled = the ablation's schedule changes that component within the run's R = {ctx.R} rounds.",
        *(
            [
                "Partly zero by construction: an accept decided against the incumbent's own ground-truth "
                "generations (greedy decoding) cannot be a false accept. Accepts of that kind behind the shown "
                "false-accept rate (pooled over seeds): " + "; ".join(part) + "."
            ]
            if part
            else []
        ),
        *(
            []
            if has_coupling
            else [
                "The ablation results do not record which accepts are coupled to the ground truth: under greedy "
                "decoding an accept decided against the incumbent's own cached generation cannot be a false "
                "accept, so these rates can be partly zero by construction (Table 3, extended, counts such "
                "accepts for the main schedule)."
            ]
        ),
        *_construction_notes(cells),
        *ctx.skipped_note(),
    ]
    return _finish(
        ctx,
        "T8",
        _frame(rows, PAPER_COLUMNS["T8"]),
        _frame(ext_rows, ext_cols) if ext_rows else None,
        notes,
        caption="Which environment changes and which reference-age behaviour drive inflation and false accepts.",
    )


# --------------------------------------------------------------------------- T9


def per_seed_policy_frame(bundle: AnalysisBundle) -> pd.DataFrame:
    """One row per (policy, seed): ``gt_acc``, ``far``, ``n_accepted``, ``n_false_accepts``,
    ``n_accepted_gt_coupled``, ``calls_total``, ``refreshes``, ``final_inc``.

    Uses the bundle's ``policy_per_seed`` frame when present; otherwise derives it from ``policy_rounds``
    (decisions, calls), ``policy_summary`` (initial-reference calls, which are not charged to any round) and
    ``accuracy`` (final incumbent's ground-truth accuracy under the canonical env at the last round).
    """
    cols = [
        "policy",
        "seed",
        "gt_acc",
        "far",
        "n_accepted",
        "n_false_accepts",
        "n_accepted_gt_coupled",
        "calls_total",
        "refreshes",
        "final_inc",
    ]
    ready = bundle.frame("policy_per_seed")
    if not ready.empty and {"policy", "seed", "gt_acc", "far", "calls_total"} <= set(ready.columns):
        out = ready.copy()
        for c in cols:
            if c not in out.columns:
                out[c] = np.nan
        return out[cols]
    pr = bundle.frame("policy_rounds")
    need = {"policy", "seed", "round", "accepted", "false_accept"}
    if pr.empty or not need <= set(pr.columns):
        return pd.DataFrame(columns=cols)
    plan = _plan_from_meta(bundle.meta.get("plan"))
    canon = str(plan.gt.canonical_env)
    acc = bundle.frame("accuracy")
    have_acc = not acc.empty and {"seed", "env", "slot", "round", "gt_acc"} <= set(acc.columns)
    summ = bundle.frame("policy_summary")
    ref_init: dict[str, float] = {}
    if not summ.empty and {"policy", "calls_reference_init"} <= set(summ.columns):
        ref_init = {
            str(p): _num(v) for p, v in zip(summ["policy"], summ["calls_reference_init"], strict=False)
        }
    rows = []
    for (pol, seed), g in pr.groupby([pr["policy"].astype(str), pd.to_numeric(pr["seed"])], sort=False):
        g = g.sort_values("round")
        accepted = g["accepted"].map(_truthy)
        n_acc = int(accepted.sum())
        n_fa = int(g["false_accept"].map(_truthy).sum())
        coupled = int((accepted & g["gt_coupled"].map(_truthy)).sum()) if "gt_coupled" in g.columns else 0
        calls = float("nan")
        if "calls_round_total" in g.columns:
            init = ref_init.get(str(pol), 0.0)
            calls = float(pd.to_numeric(g["calls_round_total"], errors="coerce").sum()) + (
                init if math.isfinite(init) else 0.0
            )
        refreshes = (
            int(g["refresh_kind"].astype(str).isin(["refresh", "fixed_age"]).sum())
            if "refresh_kind" in g.columns
            else np.nan
        )
        final_inc = _int(g["inc_after"].iloc[-1], -1) if "inc_after" in g.columns else -1
        last = _int(g["round"].max())
        gt = float("nan")
        if have_acc and final_inc >= 0:
            m = (
                (pd.to_numeric(acc["seed"]) == int(seed))
                & (acc["env"].astype(str) == canon)
                & (pd.to_numeric(acc["slot"]) == final_inc)
                & (pd.to_numeric(acc["round"]) == last)
            )
            hit = acc.loc[m, "gt_acc"]
            if len(hit):
                gt = _num(hit.iloc[0])
        rows.append(
            [
                str(pol),
                int(seed),
                gt,
                n_fa / n_acc if n_acc else float("nan"),
                n_acc,
                n_fa,
                coupled,
                calls,
                refreshes,
                final_inc,
            ]
        )
    return pd.DataFrame(rows, columns=cols)


def _seed_inflation(ctx: _Ctx) -> dict[int, float]:
    pairs = _pairs(ctx)
    if pairs.empty:
        return {}
    m = (pairs["env"].astype(str) == str(ctx.plan.t7.changed_env)) & (
        pd.to_numeric(pairs["age"]) == int(ctx.plan.t4.age)
    )
    sub = pairs.loc[m]
    if sub.empty:
        return {}
    vals = pd.to_numeric(sub["inflation"], errors="coerce")
    return {int(s): float(v) for s, v in vals.groupby(pd.to_numeric(sub["seed"])).mean().items()}


def _build_t9(ctx: _Ctx) -> TableSpec:
    ps = per_seed_policy_frame(ctx.bundle)
    infl = _seed_inflation(ctx)
    summ = ctx.summary_rows()
    seeds = list(ctx.seeds)
    if not seeds and not ps.empty:
        seeds = sorted({int(s) for s in ps["seed"]})
    look: dict[tuple[str, int], dict[str, Any]] = {}
    for r in ps.to_dict("records") if not ps.empty else []:
        look[(str(r["policy"]), int(r["seed"]))] = r
    rows, ext_rows = [], []
    for seed in seeds:
        for p in ctx.headline:
            r = look.get((p, int(seed)))
            label = f"{p}: {ctx.policy_label(p, summ.get(p))}"
            coupled = (
                r is not None
                and _int(r.get("n_accepted")) > 0
                and _int(r.get("n_accepted_gt_coupled")) == _int(r.get("n_accepted"))
            )
            vals = [
                fmt_pct(_get(r, "gt_acc")),
                fmt_num(infl.get(int(seed)), 2, 100),
                mark(fmt_pct(_get(r, "far")), coupled),
            ]
            rows.append([str(seed), label, *vals, fmt_int(_get(r, "calls_total"))])
            ext_rows.append(
                [
                    str(seed),
                    label,
                    *vals,
                    frac_str(_get(r, "n_false_accepts"), _get(r, "n_accepted")),
                    fmt_int(_get(r, "final_inc"))
                    if r is not None and _int(r.get("final_inc"), -1) >= 0
                    else DASH,
                    fmt_count(_get(r, "refreshes")),
                    fmt_int(_get(r, "calls_total"), True),
                ]
            )
    for p in ctx.headline:
        recs = [look[(p, int(s))] for s in seeds if (p, int(s)) in look]
        label = f"{p}: {ctx.policy_label(p, summ.get(p))}"
        n_acc = sum(_int(r.get("n_accepted")) for r in recs)
        coupled = n_acc > 0 and sum(_int(r.get("n_accepted_gt_coupled")) for r in recs) == n_acc
        calls = [_num(r.get("calls_total")) for r in recs]
        m, sd, n = mean_sd(calls)
        calls_s = DASH if n == 0 else f"{fmt_count(m)} ± {fmt_count(sd)}"
        vals = [
            pct_mean_sd([r.get("gt_acc") for r in recs]),
            pp_mean_sd([infl.get(int(s)) for s in seeds if int(s) in infl]),
            mark(pct_mean_sd([r.get("far") for r in recs]), coupled),
        ]
        rows.append(["Mean ± standard deviation", label, *vals, calls_s])
        k_fa = sum(_int(r.get("n_false_accepts")) for r in recs)
        ext_rows.append(
            [
                "Mean ± standard deviation",
                label,
                *vals,
                frac_str(k_fa, n_acc) if recs else DASH,
                DASH,
                pct_mean_sd([r.get("refreshes") for r in recs], 1, 1.0),
                DASH if n == 0 else f"{fmt_count(m, True)} ± {fmt_count(sd, True)}",
            ]
        )
    ext_cols = (
        "Seed",
        "Refresh policy",
        "Ground-truth accuracy (%)",
        "Drift inflation (pp)",
        "False-accept rate (%)",
        "False accepts / accepts",
        "Final incumbent slot",
        "Reference refreshes",
        "Model calls",
    )
    changed = str(ctx.plan.t7.changed_env)
    notes = [
        f"Drift inflation: that seed's mean inflation over candidate/reference pairs {ctx.plan.t4.age} rounds apart "
        f"under {changed} ({ctx.env_label(changed).lower()}); it does not depend on the refresh policy.",
        f"Ground-truth accuracy: final shadow incumbent under {ctx.plan.gt.canonical_env}; false-accept rate: false "
        "accepts / accepts of that seed (— without accepts; the mean row averages seeds with accepts). "
        + ctx.convention_text(),
        ctx.seed_sd_text(),
        *ctx.far_notes(ctx.headline),
        *ctx.skipped_note(),
    ]
    return _finish(
        ctx,
        "T9",
        _frame(rows, PAPER_COLUMNS["T9"]),
        _frame(ext_rows, ext_cols),
        notes,
        caption="Per-seed results of the headline policies and their spread across seeds.",
    )


# --------------------------------------------------------------------------- T10


def _build_t10(ctx: _Ctx) -> TableSpec:
    pages = ctx.app_dir / "pages"
    rows, ext_rows = [], []
    for comp, inp, op, out, page, frames in T10_COMPONENTS:
        status = "Implemented" if (pages / page).is_file() else "Planned"
        rows.append([comp, inp, op, out, status])
        counts = []
        for f in (s.strip() for s in frames.split(",")):
            if f.startswith("meta."):
                counts.append(f"{f}: {'yes' if _dig(ctx.meta, f.removeprefix('meta.')) else 'no'}")
            else:
                counts.append(f"{f}: {len(ctx.frame(f)):,} rows")
        ext_rows.append([comp, inp, op, out, status, f"app/pages/{page}", "; ".join(counts)])
    ext_cols = (*PAPER_COLUMNS["T10"], "Dashboard page", "Data in this run (bundle frames)")
    notes = [
        "Status: Implemented = the Streamlit page exists under app/pages/; Planned = not built yet. Launch the "
        "dashboard with `driftlab dashboard --run-dir <run dir>` (it reads run directories read-only).",
    ]
    return _finish(
        ctx,
        "T10",
        _frame(rows, PAPER_COLUMNS["T10"]),
        _frame(ext_rows, ext_cols),
        notes,
        caption="Components of the DriftLab demonstration and the dashboard page that shows each one.",
    )


# --------------------------------------------------------------------------- reported (teammate)


def load_reported(path: str | Path | None = None) -> dict | None:
    """The teammate's transcribed numbers (``results/reported/teammate.yaml``); ``None`` if absent/invalid."""
    import yaml

    p = Path(path) if path is not None else REPORTED_PATH
    if not p.is_file():
        return None
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError):  # ValueError: not UTF-8 text
        return None
    return data if isinstance(data, dict) else None


def _build_t3_reported(ctx: _Ctx, rep: Mapping[str, Any]) -> TableSpec | None:
    items = rep.get("T3")
    if not isinstance(items, list) or not items:
        return None
    rows = []
    for it in items:
        if not isinstance(it, Mapping):
            continue
        pol = _text(it.get("policy"))
        name = _text(it.get("name")) or PAPER_POLICIES.get(pol, (pol, ""))[0]
        rows.append(
            [
                f"{pol}: {name}" if pol else name,
                _text(it.get("refresh_rule"), DASH) or DASH,
                _text(it.get("gt_acc"), DASH) or DASH,
                _text(it.get("far"), DASH) or DASH,
                _text(it.get("model_calls"), DASH) or DASH,
                _text(it.get("refreshes"), DASH) or DASH,
            ]
        )
    notes = [n for n in (_text(rep.get("source")), _text(rep.get("T3_footnote"))) if n]
    notes.append(
        "Transcribed verbatim from the proposal; not reproduced by this run and never merged with the reproduced "
        "values of Table 3."
    )
    return _finish(
        ctx,
        "T3_reported",
        _frame(rows, PAPER_COLUMNS["T3"]),
        None,
        notes,
        caption="Teammate's reported Table 3, for side-by-side comparison only.",
        reported=True,
    )


def _build_t4_reported(ctx: _Ctx, rep: Mapping[str, Any]) -> TableSpec | None:
    items = rep.get("T4")
    if not isinstance(items, list) or not items:
        return None
    rows = []
    for it in items:
        if not isinstance(it, Mapping):
            continue
        rows.append(
            [
                _text(it.get("env"), DASH) or DASH,
                _text(it.get("age"), DASH) or DASH,
                _text(it.get("stored_win"), DASH) or DASH,
                _text(it.get("rerun_win"), DASH) or DASH,
                _text(it.get("inflation"), DASH) or DASH,
            ]
        )
    notes = ["Drift inflation = stored-reference win rate − rerun-reference win rate."]
    notes += [n for n in (_text(rep.get("T4_footnote")), _text(rep.get("source"))) if n]
    notes.append(
        "Transcribed verbatim from the proposal; not reproduced by this run and never merged with the reproduced "
        "values of Table 4."
    )
    return _finish(
        ctx,
        "T4_reported",
        _frame(rows, PAPER_COLUMNS["T4"]),
        None,
        notes,
        caption="Teammate's reported Table 4, for side-by-side comparison only.",
        reported=True,
    )


# --------------------------------------------------------------------------- build_tables

_BUILDERS: dict[str, Callable[[_Ctx], TableSpec]] = {
    "T1": _build_t1,
    "T2": _build_t2,
    "T3": _build_t3,
    "T4": _build_t4,
    "T5": _build_t5,
    "T6": _build_t6,
    "T7": _build_t7,
    "T7b": _build_t7b,
    "T8": _build_t8,
    "T9": _build_t9,
    "T10": _build_t10,
}


def _failed(ctx: _Ctx, tid: str, err: Exception) -> TableSpec:
    cols = PAPER_COLUMNS[tid]
    return _finish(
        ctx,
        tid,
        _frame([], cols),
        None,
        [f"Table could not be built from this bundle: {type(err).__name__}: {err}"],
    )


def build_tables(
    bundle: AnalysisBundle,
    *,
    reported: dict | None = None,
    app_dir: Path | None = None,
) -> dict[str, TableSpec]:
    """Every paper table of ``bundle``, keyed ``T1``..``T10`` + ``T7b`` (+ ``T3_reported`` / ``T4_reported``).

    ``reported``: the teammate's numbers; ``None`` loads :data:`REPORTED_PATH` when it exists, ``{}`` disables
    the reported tables. ``app_dir``: the dashboard directory whose ``pages/`` decide T10's status (default
    ``<repo>/app``). Keys follow :data:`TABLE_ORDER`. A builder that fails on an unexpected bundle yields a
    table of "—" with the error in its footnotes instead of aborting the others.
    """
    ctx = _Ctx(bundle, app_dir)
    rep = load_reported() if reported is None else reported
    built: dict[str, TableSpec] = {}
    for tid in TABLE_ORDER:
        if tid.endswith("_reported"):
            if not rep:
                continue
            fn = _build_t3_reported if tid == "T3_reported" else _build_t4_reported
            spec = fn(ctx, rep)
            if spec is not None:
                built[tid] = spec
            continue
        try:
            built[tid] = _BUILDERS[tid](ctx)
        except Exception as e:  # one malformed frame must not take every other table down
            built[tid] = _failed(ctx, tid, e)
    return built


def write_tables(
    bundle: AnalysisBundle,
    out_dir: str | Path,
    formats: Sequence[str] = ("md", "csv", "tex"),
    **kwargs: Any,
) -> list[Path]:
    """Alias of :func:`driftlab.reporting.render.write_tables` (the pipeline imports it from here)."""
    from driftlab.reporting.render import write_tables as _write_tables

    return _write_tables(bundle, out_dir, formats, **kwargs)


__all__ = [
    "ABLATION_LABELS",
    "DAGGER",
    "DASH",
    "PAPER_COLUMNS",
    "PAPER_POLICIES",
    "REPORTED_PATH",
    "T1_ROWS",
    "TABLE_ORDER",
    "TITLES",
    "TableSpec",
    "build_tables",
    "ci_str",
    "fmt_count",
    "fmt_int",
    "fmt_num",
    "fmt_pct",
    "frac_str",
    "is_synthetic",
    "load_reported",
    "mark",
    "mean_sd_str",
    "per_seed_policy_frame",
    "pct_mean_sd",
    "pp_mean_sd",
    "prereg_text",
    "table_label",
    "with_ci",
    "write_tables",
    "yes_no",
]
