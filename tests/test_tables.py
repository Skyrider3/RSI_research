"""Paper tables T1-T10 (+T7b, reported T3/T4) and their Markdown / CSV / LaTeX rendering.

The bundle is built by hand from a small synthetic cube through the real analysis producers (all_pairs,
summarize_pairs, table4/table7_frame, simulate_all, runs_to_frames, summarize_policies, candidates_frame,
run_ablations), so the column contract between analysis and reporting is exercised, not mocked.
"""

from __future__ import annotations

import copy
import csv
import io
import math
import re

import numpy as np
import pandas as pd
import pytest

from driftlab.analysis.ablations import run_ablations
from driftlab.analysis.allpairs import all_pairs, summarize_pairs, table4_frame, table7_frame
from driftlab.analysis.bundle_io import (
    BY_CONSTRUCTION_NOTE,
    SYNTHETIC_NOTE,
    AnalysisBundle,
    load_bundle,
    save_bundle,
)
from driftlab.analysis.cube import empty_cube, make_trajectory, set_cell
from driftlab.analysis.policies import (
    candidates_frame,
    per_seed_frame,
    runs_to_frames,
    simulate_all,
    summarize_policies,
)
from driftlab.config import REPO_ROOT, ExperimentConfig, load_config, load_plan
from driftlab.environments import build_environments, diff
from driftlab.extraction import extractor_tags
from driftlab.keys import rng_seed
from driftlab.reporting import tables as tables_mod
from driftlab.reporting.render import (
    ALL_TABLES,
    ALL_TABLES_EXTENDED,
    has_results_component,
    latex_escape,
    render_csv,
    render_latex,
    render_markdown,
    write_tables,
)
from driftlab.reporting.tables import (
    DASH,
    TABLE_ORDER,
    build_tables,
    ci_str,
    fmt_int,
    fmt_num,
    load_reported,
    mark,
    pct_mean_sd,
    per_seed_policy_frame,
    pp_mean_sd,
    with_ci,
)

R, N, SEEDS = 4, 20, (0, 1)
PLAN = load_plan(REPO_ROOT / "analysis_plans" / "prereg_v1.yaml")
ENVS = build_environments()
TAGS = extractor_tags()
NO_REPORTED: dict = {}

EXPECTED_HEADERS = {
    "T1": ["Component", "Configuration"],
    "T2": ["Condition ID", "Environment condition", "Reference version", "Current environment", "Purpose"],
    "T3": [
        "Policy",
        "Reference refresh rule",
        "Ground-truth accuracy (%) ↑",
        "False-accept rate (%) ↓",
        "Model calls ↓",
        "Reference refreshes ↓",
    ],
    "T4": [
        "Environment condition",
        "Reference age (rounds)",
        "Stored-reference win rate (%)",
        "Rerun-reference win rate (%)",
        "Drift inflation (percentage points)",
    ],
    "T5": [
        "Candidate ID",
        "Seed",
        "Incumbent accuracy (%)",
        "Candidate accuracy (%)",
        "Accuracy change (pp)",
        "Ground-truth outcome",
    ],
    "T6": [
        "Refresh policy",
        "Candidates evaluated",
        "Candidates accepted",
        "Accepted candidates with no ground-truth improvement",
        "False-accept rate (%)",
    ],
    "T7": [
        "Reference age (optimization rounds)",
        "Environment status",
        "Stored-reference win rate (%)",
        "Rerun-reference win rate (%)",
        "Drift inflation (pp)",
        "False-accept rate (%)",
    ],
    "T8": [
        "Ablation",
        "Decoding changes enabled",
        "Extraction changes enabled",
        "Reference-age variation enabled",
        "Drift inflation (pp)",
        "False-accept rate (%)",
    ],
    "T9": [
        "Seed",
        "Refresh policy",
        "Ground-truth accuracy (%)",
        "Drift inflation (pp)",
        "False-accept rate (%)",
        "Model calls",
    ],
    "T10": ["Demonstration component", "Input", "System operation", "Output shown to user", "Status"],
}
T1_ROWS = [
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
]
T10_ROWS = [
    "Reference manager",
    "Environment version tracker",
    "Reference rerun mechanism",
    "Candidate comparison view",
    "Ground-truth evaluation panel",
    "False-accept and drift dashboard",
    "Refresh-policy comparison",
    "Reproducibility / experiment logs",
]


# --------------------------------------------------------------------------- fixtures


def _cube():
    """Full-triangle cube: greedy reruns are cache hits of the creation generation except at ages 1 and 3
    (physical, with a few flips), every t02 round is an independent draw, one ("gt", 0) draw per slot; v2 is a
    superset of v1 (lenient)."""
    draws = [("round", r) for r in range(R + 1)] + [("gt", 0)]
    cube = empty_cube("tables-test", SEEDS, ["greedy", "t02"], ["v1", "v2"], R + 1, draws, N)
    cube.synthetic = True
    for s in SEEDS:
        rng = np.random.default_rng(rng_seed("test-tables", s))
        quality = rng.uniform(0.35, 0.8, size=R + 1)

        def outputs(q: float, rng: np.random.Generator = rng) -> dict[str, np.ndarray]:
            v1 = rng.random(N) < q
            return {"v1": v1, "v2": v1 | (rng.random(N) < 0.25)}

        for k in range(R + 1):
            created = outputs(quality[k])
            set_cell(cube, s, "greedy", k, ("round", k), created, physical=True)
            rows = cube.gen_rows(s, "greedy", k, ("round", k)).copy()
            for r in range(k + 1, R + 1):
                if r - k in (1, 3):
                    flip = rng.random(N) < 0.1
                    w1 = created["v1"] ^ flip
                    w2 = (created["v2"] ^ flip) | w1
                    set_cell(cube, s, "greedy", k, ("round", r), {"v1": w1, "v2": w2}, physical=True)
                else:
                    set_cell(cube, s, "greedy", k, ("round", r), created, gen_rows=rows)
            for r in range(k, R + 1):
                set_cell(cube, s, "t02", k, ("round", r), outputs(quality[k]), physical=True)
            set_cell(cube, s, "t02", k, ("gt", 0), outputs(quality[k]), physical=True)
            set_cell(cube, s, "greedy", k, ("gt", 0), created, gen_rows=rows)
    return cube


def _environments_frame() -> pd.DataFrame:
    st = ENVS["E1"]
    rows = []
    for eid, e in ENVS.items():
        d = e.decoding
        rows.append(
            {
                "env_id": eid,
                "decoding": d.id,
                "temperature": d.temperature,
                "extractor": e.extractor,
                "extractor_tag": TAGS[e.extractor],
                "fingerprint": e.fingerprint(TAGS),
                "description": e.description,
                "changed_vs_storage": ",".join(diff(st, e, TAGS)),
                "top_p": d.top_p,
                "top_k": d.top_k,
                "repetition_penalty": d.repetition_penalty,
                "max_new_tokens": d.max_new_tokens,
                "is_storage_env": eid == "E1",
            }
        )
    return pd.DataFrame(rows)


def _accuracy_frame(cube) -> pd.DataFrame:
    rows = []
    for s in SEEDS:
        for eid, e in ENVS.items():
            for k in range(R + 1):
                for r in range(k, R + 1):
                    acc = cube.gt_acc(s, e.decoding.id, k, r, e.extractor, mode=PLAN.gt.mode)
                    rows.append({"seed": s, "env": eid, "slot": k, "round": r, "gt_acc": acc})
    return pd.DataFrame(rows)


def _config() -> ExperimentConfig:
    return load_config(
        REPO_ROOT / "configs" / "smoke_mock.yaml",
        overrides=[f"run.rounds={R}", f"data.eval.n={N}", f"data.dev.n={N}", "run.name=tables_test"],
    )


@pytest.fixture(scope="module")
def built():
    cube = _cube()
    trajs = {
        s: make_trajectory(s, [f"s{s}-prompt-{k}" for k in range(R + 1)], adv, [0.5] * (R + 1))
        for s, adv in zip(
            SEEDS, ([False, True, False, True, True], [False, False, True, True, False]), strict=True
        )
    }
    pr = all_pairs(cube, trajs, ENVS, PLAN)
    summary = summarize_pairs(pr, by=("env", "age"), B=50, seed=PLAN.bootstrap.seed)
    names = [*PLAN.policies, "FIXEDAGE_k1"]
    runs = simulate_all(cube, trajs, PLAN, ENVS, N, TAGS, policies=names)
    rounds_df, refs_df = runs_to_frames(runs)
    cfg = _config()
    frames = {
        "pairs": pr.df,
        "pair_summary": summary,
        "table4": table4_frame(summary, PLAN),
        "table7": table7_frame(summary, PLAN),
        "policy_rounds": rounds_df,
        "policy_refs": refs_df,
        "policy_summary": summarize_policies(runs),
        "candidates": candidates_frame(cube, trajs, PLAN, ENVS, runs, extractor_tags=TAGS),
        "ablations": run_ablations(cube, trajs, PLAN, ENVS, pr.df, N, TAGS),
        "environments": _environments_frame(),
        "accuracy": _accuracy_frame(cube),
        "ledger_summary": pd.DataFrame(
            [
                {
                    "stage": "matrix",
                    "purpose": "eval_matrix",
                    "seed": 0,
                    "n_requested": 300,
                    "n_executed": 250,
                },
                {
                    "stage": "matrix",
                    "purpose": "eval_matrix",
                    "seed": 1,
                    "n_requested": 300,
                    "n_executed": 240,
                },
            ]
        ),
    }
    meta = {
        "run_id": "tables_test",
        "run_dir": "unused",
        "config_hash": cfg.config_hash(),
        "plan_hash": PLAN.plan_hash(),
        "plan_version": PLAN.version,
        "synthetic": True,
        "created_at": "2026-10-05T00:00:00Z",
        "seeds": list(SEEDS),
        "R": R,
        "N": N,
        "D": N,
        "schedule": {str(k): v for k, v in PLAN.schedule.items()},
        "extractor_tags": dict(TAGS),
        "engine_fp": "mockengine000000",
        "config": cfg.model_dump(mode="json"),
        "plan": PLAN.model_dump(mode="json"),
        "provenance": {"python": {"version": "3.11.0"}, "packages": {"numpy": "2.0", "pandas": "3.0"}},
        "driftlab_version": "0.1.0",
        "warnings": [],
        "skipped_pairs": 0,
        "B": 50,
    }
    bundle = AnalysisBundle(meta=meta, frames=frames)
    return bundle, runs


@pytest.fixture(scope="module")
def bundle(built) -> AnalysisBundle:
    return built[0]


@pytest.fixture(scope="module")
def specs(bundle):
    return build_tables(bundle, reported=NO_REPORTED)


def _real(bundle: AnalysisBundle) -> AnalysisBundle:
    """The same frames presented as a real (vLLM) run: no synthetic flag and no mock backend anywhere."""
    meta = copy.deepcopy(bundle.meta)
    meta["synthetic"] = False
    meta["config"]["backend"]["kind"] = "vllm"
    return AnalysisBundle(meta=meta, frames=bundle.frames)


def _with(bundle: AnalysisBundle, meta: dict | None = None, **frames: pd.DataFrame) -> AnalysisBundle:
    """A copy of ``bundle`` with some meta keys and frames replaced."""
    new_meta = copy.deepcopy(bundle.meta)
    new_meta.update(meta or {})
    return AnalysisBundle(meta=new_meta, frames={**bundle.frames, **frames})


# --------------------------------------------------------------------------- formatting helpers


def test_formatting_helpers():
    assert pct_mean_sd([0.69, 0.73]) == "71.0 ± 2.8"
    assert pct_mean_sd([0.71]) == "71.0 ± 0.0"
    assert pct_mean_sd([float("nan"), None]) == DASH
    assert pct_mean_sd([0.5, float("nan")]) == "50.0 ± 0.0"  # NaN-aware (seeds without accepts)
    assert pp_mean_sd([0.0906, 0.0906]) == "9.06 ± 0.00"
    assert pp_mean_sd([0.02, 0.03], digits=1) == "2.5 ± 0.7"
    assert ci_str(0.011, 0.034) == "[1.10, 3.40]"
    assert ci_str(float("nan"), 0.1) == DASH
    assert with_ci(0.0227, 0.011, 0.034) == "2.27 [1.10, 3.40]"
    assert with_ci(0.0227, None, None) == "2.27"
    assert with_ci(None, 0.0, 0.1) == DASH
    assert fmt_int(4411) == "4411" and fmt_int(4411.4) == "4411"
    assert fmt_int(4411, thousands=True) == "4,411"
    assert fmt_int(float("nan")) == DASH
    assert fmt_num(-0.00001, 2, 100) == "0.00"  # no "-0.00"
    assert fmt_num(0.208333, 1, 100, signed=True) == "+20.8"
    assert fmt_num("True") == "1.0" and fmt_num("abc") == DASH
    assert mark("0.00", True) == "0.00 ‡" and mark(DASH, True) == DASH and mark("1.0", False) == "1.0"


# --------------------------------------------------------------------------- structure


def test_keys_follow_table_order(bundle, specs):
    assert list(specs) == [t for t in TABLE_ORDER if not t.endswith("_reported")]
    assert set(specs) == {f"T{i}" for i in range(1, 11)} | {"T7b"}
    with_rep = build_tables(bundle)  # reported=None loads results/reported/teammate.yaml when it exists
    if load_reported() is not None:
        assert list(with_rep) == list(TABLE_ORDER)


@pytest.mark.parametrize("tid", list(EXPECTED_HEADERS))
def test_paper_headers_exact(specs, tid):
    assert list(specs[tid].df.columns) == EXPECTED_HEADERS[tid]
    assert (
        specs[tid].title
        == {
            "T1": "Experimental configuration",
            "T2": "Environment conditions",
            "T3": "Main comparison of reference-refresh policies",
            "T4": "Environment-drift inflation",
            "T5": "Ground-truth evaluation of candidate prompts",
            "T6": "False-accept analysis",
            "T7": "Effect of reference age",
            "T8": "Ablation study",
            "T9": "Reproducibility across random seeds",
            "T10": "System demonstration functionality",
        }[tid]
    )


def test_no_builder_failed(specs):
    for tid, spec in specs.items():
        assert not any("could not be built" in n for n in spec.footnotes), (tid, spec.footnotes)
        assert spec.extended is not None, tid
        assert all(isinstance(v, str) for v in spec.df.to_numpy().ravel()), tid


def test_t1_rows(specs):
    df = specs["T1"].df
    assert df["Component"].tolist() == T1_ROWS
    vals = dict(zip(df["Component"], df["Configuration"], strict=True))
    assert vals["Target model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert vals["Model checkpoint / revision"] == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    assert vals["Development examples"].startswith(f"First {N} questions of the train split")
    assert vals["Evaluation examples"].startswith(f"First {N} questions of the test split")
    assert vals["Baseline decoding"].startswith("Greedy (temperature 0")
    assert "640 new tokens" in vals["Baseline decoding"]
    assert vals["Alternative decoding"].startswith("Temperature 0.2 sampling")
    assert (
        TAGS["v1"] in vals["Answer extraction versions"] and TAGS["v2"] in vals["Answer extraction versions"]
    )
    assert vals["Number of random seeds"] == "2 (seeds 0, 1)"
    assert "numpy 2.0" in vals["Software / package versions"]
    ext = specs["T1"].extended
    ext_vals = dict(zip(ext["Component"], ext["Configuration"], strict=True))
    for key in (
        "Rounds per seed (R)",
        "Promotion rule",
        "Analysis plan hash",
        "Config hash",
        "GPU",
        "Synthetic data",
    ):
        assert key in ext_vals
    assert ext_vals["Analysis plan hash"].startswith(PLAN.plan_hash())
    assert ext_vals["Synthetic data"].startswith("Yes")


def test_t2_rows(specs):
    df = specs["T2"].df
    assert df["Condition ID"].tolist() == ["E1", "E2", "E3", "E4"]
    assert df["Environment condition"].tolist() == [
        "Unchanged environment",
        "Decoding change",
        "Answer-extraction change",
        "Multiple environment changes",
    ]
    assert df["Purpose"].tolist() == [
        "Control: measures the null (no drift)",
        "Isolates decoding drift",
        "Isolates scoring drift",
        "Combined drift",
    ]
    tag1, tag2 = TAGS["v1"], TAGS["v2"]
    assert set(df["Reference version"]) == {f"E1: greedy + strict {tag1}, stored, scores kept"}
    assert df["Current environment"].tolist() == [
        f"greedy + strict {tag1}",
        f"T=0.2 + strict {tag1}",
        f"greedy + lenient {tag2}",
        f"T=0.2 + lenient {tag2}",
    ]


def test_t3_values_match_policy_summary(bundle, specs):
    df = specs["T3"].df
    assert df["Policy"].tolist() == [
        "P1: Frozen reference",
        "P2: Per-batch refresh",
        "P3: Environment-triggered refresh",
    ]
    assert df["Reference refresh rule"].tolist() == [
        "Never refresh",
        "Refresh every batch",
        "Refresh after detected environment changes",
    ]
    summ = bundle.frame("policy_summary").set_index("policy")
    for row in df.to_dict("records"):
        p = row["Policy"].split(":")[0]
        s = summ.loc[p]
        assert row["Ground-truth accuracy (%) ↑"] == f"{s.gt_acc_mean * 100:.1f} ± {s.gt_acc_sd * 100:.1f}"
        assert re.fullmatch(r"\d+", row["Model calls ↓"]), row  # plain integer, no separator
        assert int(row["Model calls ↓"]) == round(s.calls_total)
        assert row["Reference refreshes ↓"] == str(int(s.refreshes))
        far = row["False-accept rate (%) ↓"]
        if math.isfinite(s.far_seed_mean):
            assert far.startswith(f"{s.far_seed_mean * 100:.1f} ± {s.far_seed_sd * 100:.1f}")
        else:
            assert far == DASH
    # teammate_v1: R * (D + 1 + N) (+ N per refresh)
    base = R * (N + 1 + N)
    assert df.loc[0, "Model calls ↓"] == str(base)
    assert int(df.loc[1, "Model calls ↓"]) == base + R * N
    ext = specs["T3"].extended
    assert len(ext) == len(summ) and "Executed calls (whole run, shared)" in ext.columns
    assert set(ext["Executed calls (whole run, shared)"]) == {"490"}
    assert any("Wilson" in c for c in ext.columns)


def test_t3_footnote_mirrors_teammate(specs):
    first = specs["T3"].footnotes[0]
    assert first.startswith(
        f"Mean ± standard deviation over {len(SEEDS)} seeds ({N} test questions, {R} rounds each)."
    )
    assert "False-accept rate is worked out per seed and then averaged" in first
    assert "The frozen reference is the stored output of the round-0 prompt" in first


def test_t4_rows_and_format(bundle, specs):
    df = specs["T4"].df
    assert df["Environment condition"].tolist() == [
        "Unchanged",
        "Decoding change",
        "Extraction change",
        "Multiple changes",
    ]
    assert set(df["Reference age (rounds)"]) == {str(PLAN.t4.age)}
    pat = re.compile(r"-?\d+\.\d\d ± \d+\.\d\d( ‡)?")
    for col in EXPECTED_HEADERS["T4"][2:]:
        assert all(pat.fullmatch(v) for v in df[col]), (col, df[col].tolist())
    t4 = bundle.frame("table4").set_index("env")
    e4 = df.iloc[3]
    assert e4["Drift inflation (percentage points)"].startswith(
        f"{t4.loc['E4', 'infl_seed_mean'] * 100:.2f} ± {t4.loc['E4', 'infl_seed_sd'] * 100:.2f}"
    )
    notes = specs["T4"].footnotes
    assert notes[0] == "Drift inflation = stored-reference win rate − rerun-reference win rate."
    assert notes[1] == (
        f"Each cell averages every candidate and reference pair that are {PLAN.t4.age} rounds apart. The reference "
        "answers were stored under E1 and kept the scores they had at the time."
    )
    ext = specs["T4"].extended
    assert {
        "Pooled drift inflation (pp) [95% CI]",
        "Extraction part (pp)",
        "Generation part (pp)",
        "Pairs",
    } <= set(ext.columns)
    # E1/E2 keep the storage extractor: their extraction part is 0 by construction.
    assert ext["Extraction part (pp)"].iloc[0].endswith("‡") and ext["Extraction part (pp)"].iloc[1].endswith(
        "‡"
    )


def test_t5_one_row_per_candidate(bundle, specs):
    df = specs["T5"].df
    assert len(df) == len(SEEDS) * R
    assert df["Candidate ID"].tolist() == [f"s{s}-r{t}" for s in SEEDS for t in range(1, R + 1)]
    assert set(df["Ground-truth outcome"]) <= {"Improvement", "No change", "Regression"}
    cand = bundle.frame("candidates")
    assert df["Incumbent accuracy (%)"].iloc[0] == f"{cand['incumbent_acc'].iloc[0] * 100:.1f}"
    ext = specs["T5"].extended
    assert {"Round", "Environment at round", "Dev accuracy change (pp)", "Advanced on dev"} <= set(
        ext.columns
    )
    for p in ("P1", "P2", "P3", "ORACLE"):
        assert f"{p} decision" in ext.columns
        assert set(ext[f"{p} decision"]) <= {"Accept", "Reject"}


def test_t6_counts(bundle, specs):
    df = specs["T6"].df
    assert df["Refresh policy"].tolist() == [
        "Frozen reference",
        "Per-batch refresh",
        "Environment-triggered refresh",
    ]
    summ = bundle.frame("policy_summary").set_index("policy")
    for p, row in zip(("P1", "P2", "P3"), df.to_dict("records"), strict=True):
        assert row["Candidates evaluated"] == str(len(SEEDS) * R)
        assert row["Candidates accepted"] == str(summ.loc[p, "n_accepted"])
        assert row["Accepted candidates with no ground-truth improvement"] == str(
            summ.loc[p, "n_false_accepts"]
        )


def test_t7_rows_dagger_and_missing_ages(bundle, specs):
    df = specs["T7"].df
    labels = [(int(a), s) for a, s in zip(df.iloc[:, 0], df["Environment status"], strict=True)]
    assert labels == [(a, s) for a in (0, 1, 3, 5, 10) for s in ("Unchanged (E1)", "Changed (E4)")]
    age0 = df.iloc[0]
    # Age 0 under the storage env: the rerun IS the stored generation -> identical by construction.
    assert age0["Drift inflation (pp)"] == "0.00 ‡" and age0["Rerun-reference win rate (%)"].endswith("‡")
    # R = 4: ages 5 and 10 have no pairs -> every value "—", row kept.
    for i in range(6, 10):
        assert df.iloc[i, 2:].tolist() == [DASH] * 4
    notes = specs["T7"].footnotes
    assert any(n.startswith("Age 0:") for n in notes)
    assert any(
        n.startswith("No candidate/reference pair at age 5, 10") and "have 4 rounds" in n for n in notes
    ), notes
    assert BY_CONSTRUCTION_NOTE in notes
    # FAR column: "rate (k/n)" from the stored-reference decisions.
    t7 = bundle.frame("table7")
    e4 = t7[(t7["age"] == 1) & (t7["env"] == "E4")].iloc[0]
    far = df.iloc[3]["False-accept rate (%)"]
    if e4["n_accept_stored"]:
        assert far.startswith(f"{e4['far_cur_stored'] * 100:.1f} (") and far.endswith(
            f"/{e4['n_accept_stored']})"
        )
    ext = specs["T7"].extended
    assert {
        "False-accept rate, rerun reference (%) (k/n)",
        "False-accept rate, fresh reference (%) (k/n)",
    } <= set(ext.columns)
    assert "Decision flip rate (%)" in ext.columns and "Pairs" in ext.columns


def test_t7b_full_grid(specs):
    df = specs["T7b"].df
    assert len(df) == len(sorted({*range(R + 1), 5, 10})) * 4
    assert df["Environment"].iloc[:4].tolist() == [
        "E1 (Unchanged)",
        "E2 (Decoding change)",
        "E3 (Extraction change)",
        "E4 (Multiple changes)",
    ]
    assert "[" in df["Drift inflation (pp) [95% CI]"].iloc[5]  # age 1, E2: measured, with a CI


def test_t8_far_column_choice(bundle, specs):
    df = specs["T8"].df
    assert df["Ablation"].tolist() == [
        "A1: Full system",
        "A2: No decoding changes",
        "A3: No extraction changes",
        "A4: Fixed reference age",
    ]
    abl = bundle.frame("ablations").set_index("ablation")
    for name, row in zip(("A1", "A2", "A3", "A4"), df.to_dict("records"), strict=True):
        for col, key in (
            ("Decoding changes enabled", "decoding_changes"),
            ("Extraction changes enabled", "extraction_changes"),
            ("Reference-age variation enabled", "reference_age_variation"),
        ):
            assert row[col] == ("Yes" if abl.loc[name, key] else "No")
        far = abl.loc[name, "far_P3"] if name == "A4" else abl.loc[name, "far_P1b"]
        assert row["False-accept rate (%)"] == (DASH if not math.isfinite(far) else f"{far * 100:.1f}")
    assert df.iloc[3]["Reference-age variation enabled"] == "No"


def test_t9_derivation_matches_policy_runs(built, specs):
    bundle, runs = built
    derived = per_seed_policy_frame(bundle).set_index(["policy", "seed"])
    truth = per_seed_frame(runs).set_index(["policy", "seed"])
    for key, t in truth.iterrows():
        d = derived.loc[key]
        assert d["gt_acc"] == pytest.approx(t["gt_acc"])
        assert d["calls_total"] == pytest.approx(t["calls_total"])
        assert d["n_accepted"] == t["n_accepted"] and d["n_false_accepts"] == t["n_false_accepts"]
        assert (math.isnan(d["far"]) and math.isnan(t["far"])) or d["far"] == pytest.approx(t["far"])
        assert d["refreshes"] == t["refreshes"]
    df = specs["T9"].df
    assert len(df) == len(SEEDS) * 3 + 3
    assert df["Seed"].tolist()[-3:] == ["Mean ± standard deviation"] * 3
    assert df["Seed"].tolist()[:3] == ["0", "0", "0"]
    # Drift inflation per seed: mean inflation at age t4.age under the changed env (E4), policy-independent.
    pairs = bundle.frame("pairs")
    sub = pairs[(pairs["env"] == "E4") & (pairs["age"] == PLAN.t4.age) & (pairs["seed"] == 0)]
    assert set(df.iloc[:3]["Drift inflation (pp)"]) == {f"{sub['inflation'].mean() * 100:.2f}"}
    # A bundle that already carries per-seed values is used as is.
    ready = per_seed_frame(runs)
    ready["gt_acc"] = 0.5
    b2 = AnalysisBundle(meta=bundle.meta, frames={**bundle.frames, "policy_per_seed": ready})
    t9 = build_tables(b2, reported=NO_REPORTED)["T9"].df
    assert set(t9.iloc[: len(SEEDS) * 3]["Ground-truth accuracy (%)"]) == {"50.0"}


def test_t10_status_from_app_dir(bundle, tmp_path):
    pages = tmp_path / "app" / "pages"
    pages.mkdir(parents=True)
    (pages / "6_Drift_Dashboard.py").write_text("# page\n")
    df = build_tables(bundle, reported=NO_REPORTED, app_dir=tmp_path / "app")["T10"].df
    assert df["Demonstration component"].tolist() == T10_ROWS
    status = dict(zip(df["Demonstration component"], df["Status"], strict=True))
    assert status["False-accept and drift dashboard"] == "Implemented"
    assert {v for k, v in status.items() if k != "False-accept and drift dashboard"} == {"Planned"}


# --------------------------------------------------------------------------- footnotes


def test_every_table_ends_with_provenance_and_synthetic_note(bundle, specs):
    for tid, spec in specs.items():
        for ext in (False, True):
            notes = spec.notes(ext)
            assert notes[-1] == SYNTHETIC_NOTE, (tid, ext)
            dagger = spec.has_dagger(ext)
            prov = notes[-3] if dagger else notes[-2]
            assert PLAN.plan_hash() in prov and bundle.meta["config_hash"] in prov and "tables_test" in prov
            assert (BY_CONSTRUCTION_NOTE in notes) == dagger, (tid, ext)
            if dagger:
                assert notes[-2] == BY_CONSTRUCTION_NOTE


def test_real_bundle_has_no_synthetic_note(bundle):
    for tid, spec in build_tables(_real(bundle), reported=NO_REPORTED).items():
        assert all(SYNTHETIC_NOTE not in n for n in spec.notes(False) + spec.notes(True)), tid


def test_dagger_tables(specs):
    assert specs["T7"].has_dagger() and specs["T7b"].has_dagger()
    assert not specs["T1"].has_dagger() and BY_CONSTRUCTION_NOTE not in specs["T1"].footnotes


# --------------------------------------------------------------------------- robustness


def test_empty_bundle_renders_dashes():
    empty = AnalysisBundle(meta={}, frames={})
    out = build_tables(empty, reported=NO_REPORTED)
    assert set(out) == {f"T{i}" for i in range(1, 11)} | {"T7b"}
    for tid, spec in out.items():
        if tid in EXPECTED_HEADERS:
            assert list(spec.df.columns) == EXPECTED_HEADERS[tid]
        assert len(spec.df) >= 1
        assert not any("could not be built" in n for n in spec.footnotes), (tid, spec.footnotes)
        render_markdown(spec), render_csv(spec), render_latex(spec)
        render_markdown(spec, extended=True), render_latex(spec, extended=True)
    assert DASH in out["T3"].df["Ground-truth accuracy (%) ↑"].tolist()
    assert out["T7"].df.iloc[:, 2:].stack().eq(DASH).all()
    assert len(out["T4"].df) == 4 and len(out["T8"].df) == 4


def test_empty_frames_with_columns_do_not_crash(bundle):
    frames = {name: df.iloc[0:0] for name, df in bundle.frames.items()}
    out = build_tables(AnalysisBundle(meta=bundle.meta, frames=frames), reported=NO_REPORTED)
    for tid, spec in out.items():
        assert not any("could not be built" in n for n in spec.footnotes), (tid, spec.footnotes)
    assert out["T5"].df.iloc[0].tolist() == [DASH] * 6


def test_failing_builder_is_isolated(bundle, monkeypatch):
    def boom(ctx):
        raise RuntimeError("broken frame")

    monkeypatch.setitem(tables_mod._BUILDERS, "T6", boom)
    out = build_tables(bundle, reported=NO_REPORTED)
    assert list(out["T6"].df.columns) == EXPECTED_HEADERS["T6"]
    assert any("could not be built" in n and "broken frame" in n for n in out["T6"].footnotes)
    assert not any("could not be built" in n for n in out["T3"].footnotes)


def test_csv_round_trip_gives_identical_tables(bundle, specs, tmp_path):
    save_bundle(bundle, tmp_path / "run")
    loaded = load_bundle(tmp_path / "run")
    again = build_tables(loaded, reported=NO_REPORTED)
    for tid in specs:
        pd.testing.assert_frame_equal(again[tid].df, specs[tid].df, obj=tid)
        assert again[tid].footnotes == specs[tid].footnotes, tid


# --------------------------------------------------------------------------- rendering


def test_render_markdown(specs):
    md = render_markdown(specs["T3"])
    lines = md.splitlines()
    assert lines[0] == "### Table 3. Main comparison of reference-refresh policies"
    assert lines[2] == "| " + " | ".join(EXPECTED_HEADERS["T3"]) + " |"
    assert re.fullmatch(r"\|(---:?\|)+", lines[3])
    assert "1. Mean ± standard deviation" in md
    assert md.rstrip().endswith("*") and specs["T3"].caption in md
    ext = render_markdown(specs["T3"], extended=True)
    assert "(extended)" in ext.splitlines()[0] and "Executed calls" in ext
    assert render_markdown(specs["T7b"]).startswith("### Table 7b. ")


def test_markdown_escapes_pipes(bundle):
    spec = build_tables(bundle, reported=NO_REPORTED)["T1"]
    spec.df.iloc[0, 1] = "a | b"
    assert "a \\| b" in render_markdown(spec)


def test_render_csv(bundle, specs):
    text = render_csv(specs["T4"], mark_synthetic=False)
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == EXPECTED_HEADERS["T4"] and len(rows) == 5
    ext_rows = list(csv.reader(io.StringIO(render_csv(specs["T4"], extended=True, mark_synthetic=False))))
    assert ext_rows[0] == list(specs["T4"].extended.columns)
    real = build_tables(_real(bundle), reported=NO_REPORTED)["T4"]
    assert render_csv(real) == text  # a real run's CSV is plain data


def test_synthetic_csv_starts_with_a_marker_line(bundle, specs, tmp_path):
    """A synthetic table's CSV must stay marked when it leaves the run directory."""
    text = render_csv(specs["T3"])
    first, rest = text.split("\n", 1)
    assert first == f"# {SYNTHETIC_NOTE}"
    assert list(csv.reader(io.StringIO(rest)))[0] == EXPECTED_HEADERS["T3"]
    path = tmp_path / "T3.csv"
    path.write_text(text, encoding="utf-8")
    back = pd.read_csv(path, comment="#", dtype=str, keep_default_na=False)
    assert list(back.columns) == EXPECTED_HEADERS["T3"] and back.equals(specs["T3"].df.astype(str))
    rep = build_tables(bundle, reported=load_reported() or {"T3": [{"policy": "P1", "gt_acc": "1"}]})
    marker = render_csv(rep["T3_reported"]).split("\n", 1)[0]
    assert marker.startswith(f"# {SYNTHETIC_NOTE}") and "transcribed" in marker
    paths = write_tables(bundle, tmp_path / "out", formats=("csv",), reported=NO_REPORTED)
    assert all(p.read_text(encoding="utf-8").startswith("# SYNTHETIC DATA") for p in paths)


def test_latex_escape():
    assert latex_escape("a_b & 50% #1 ‡ ± ↑ ↓ $x {y} ~ ^") == (
        r"a\_b \& 50\% \#1 \ensuremath{\ddagger} $\pm$ $\uparrow$ $\downarrow$ \$x \{y\} "
        r"\textasciitilde{} \textasciicircum{}"
    )
    assert latex_escape("C:\\path") == r"C:\textbackslash{}path"
    assert latex_escape("—") == "---" and latex_escape("−1") == "$-$1"


def test_latex_code_spans_and_markdown_angle_brackets(specs):
    # A code span becomes \texttt with '--' protected (no en dash), never literal backticks.
    assert latex_escape("run `driftlab tables --run-dir <dir>` now") == (
        r"run \texttt{driftlab tables -{}-{}run-{}dir \textless{}dir\textgreater{}} now"
    )
    assert latex_escape("a ` b") == r"a \textasciigrave{} b"
    tex = render_latex(specs["T10"])
    assert "`" not in tex and r"\texttt{driftlab dashboard -{}-{}run-{}dir" in tex
    # Markdown: "s<seed>-r<round>" would be parsed as HTML tags and vanish; code spans stay verbatim.
    md5 = render_markdown(specs["T5"])
    assert "s&lt;seed>-r&lt;round>" in md5 and "<seed>" not in md5
    md10 = render_markdown(specs["T10"])
    assert "`driftlab dashboard --run-dir <run dir>`" in md10


def test_render_latex(specs):
    tex = render_latex(specs["T7"])
    for token in ("\\begin{table}", "\\toprule", "\\midrule", "\\bottomrule", "\\end{tabular}", "\\caption{"):
        assert token in tex
    assert "\\caption{Effect of reference age}" in tex  # LaTeX numbers the table itself
    assert "\\ddagger" in tex and "(\\%)" in tex and "$\\pm$" not in tex.split("\\midrule")[0]
    for raw in ("‡", "±", "↑", "↓", "−", "—"):
        assert raw not in tex, raw
    t3 = render_latex(specs["T3"])
    assert "$\\uparrow$" in t3 and "$\\downarrow$" in t3 and "$\\pm$" in t3
    assert "teammate\\_v1" in t3  # underscores escaped in footnotes
    ext = render_latex(specs["T3"], extended=True)
    assert "\\resizebox" in ext and "tab:t3-extended" in ext


# --------------------------------------------------------------------------- writing


def test_write_tables_files_and_order(bundle, tmp_path):
    out = tmp_path / "exports" / "tables"
    paths = write_tables(bundle, out, reported=NO_REPORTED)
    names = {p.name for p in paths}
    assert all(p.exists() for p in paths)
    for tid in [f"T{i}" for i in range(1, 11)] + ["T7b"]:
        for ext in ("md", "csv", "tex"):
            assert f"{tid}.{ext}" in names and f"{tid}_extended.{ext}" in names
    assert ALL_TABLES in names and ALL_TABLES_EXTENDED in names
    text = (out / ALL_TABLES).read_text(encoding="utf-8")
    assert text.startswith("# DriftLab paper tables") and SYNTHETIC_NOTE in text.split("### ")[0]
    heads = [f"### Table {n}." for n in ("1", "2", "3", "4", "5", "6", "7", "7b", "8", "9", "10")]
    pos = [text.index(h) for h in heads]
    assert pos == sorted(pos)
    assert "(extended)" in (out / ALL_TABLES_EXTENDED).read_text(encoding="utf-8")


def test_write_tables_formats_and_extended_flag(bundle, tmp_path):
    paths = write_tables(bundle, tmp_path / "t", formats=("csv",), extended=False, reported=NO_REPORTED)
    assert paths and {p.suffix for p in paths} == {".csv"}
    assert not any("_extended" in p.name for p in paths)
    with pytest.raises(ValueError, match="format"):
        write_tables(bundle, tmp_path / "t2", formats=("pdf",))


def test_write_tables_refuses_synthetic_into_results(bundle, tmp_path):
    for target in (tmp_path / "results" / "tables", tmp_path / "Results"):
        with pytest.raises(ValueError, match="SYNTHETIC"):
            write_tables(bundle, target, reported=NO_REPORTED)
        with pytest.raises(ValueError, match="SYNTHETIC"):
            write_tables(bundle, target, reported=NO_REPORTED, allow_results_dir=True)
        assert not target.exists()
    real = write_tables(_real(bundle), tmp_path / "results" / "tables", formats=("md",), reported=NO_REPORTED)
    assert real and all(p.exists() for p in real)
    assert has_results_component("a/results/b") and not has_results_component(tmp_path / "my_results_x")


def test_write_tables_alias_used_by_the_pipeline(bundle, tmp_path):
    paths = tables_mod.write_tables(bundle, tmp_path / "x", ("md",), reported=NO_REPORTED)
    assert (tmp_path / "x" / ALL_TABLES) in paths


# --------------------------------------------------------------------------- reported (teammate)


def test_load_reported(tmp_path):
    rep = load_reported()
    if (REPO_ROOT / "results" / "reported" / "teammate.yaml").exists():
        assert rep is not None and len(rep["T3"]) == 3 and len(rep["T4"]) == 4
    assert load_reported(tmp_path / "missing.yaml") is None
    bad = tmp_path / "bad.yaml"
    bad.write_text("T3: [unclosed\n")
    assert load_reported(bad) is None
    binary = tmp_path / "binary.yaml"
    binary.write_bytes(b"\xff\xfe\x00garbage")
    assert load_reported(binary) is None


def test_reported_tables_are_separate(bundle):
    rep = {
        "source": "Proposal doc (test).",
        "T3": [{"policy": "P1", "name": "Frozen reference", "refresh_rule": "Never refresh", "gt_acc": "99.9 ± 9.9",
                "far": "88.8 ± 8.8", "model_calls": 4411, "refreshes": 0}],
        "T3_footnote": "Teammate footnote.",
        "T4": [{"env": "Unchanged", "age": 3, "stored_win": "77.77 ± 7.77", "rerun_win": "77.77 ± 7.77",
                "inflation": "0.00 ± 0.00"}],
        "T4_footnote": "Teammate T4 footnote.",
    }  # fmt: skip
    out = build_tables(bundle, reported=rep)
    assert list(out) == list(TABLE_ORDER)
    t3r, t4r = out["T3_reported"], out["T4_reported"]
    suffix = "Reported by teammate (transcribed; not reproduced by this run)"
    assert t3r.title.endswith(suffix) and t4r.title.endswith(suffix)
    assert t3r.reported and t3r.extended is None
    assert list(t3r.df.columns) == EXPECTED_HEADERS["T3"] and list(t4r.df.columns) == EXPECTED_HEADERS["T4"]
    assert t3r.df.iloc[0].tolist() == [
        "P1: Frozen reference",
        "Never refresh",
        "99.9 ± 9.9",
        "88.8 ± 8.8",
        "4411",
        "0",
    ]
    assert "Teammate footnote." in t3r.footnotes and "Proposal doc (test)." in t4r.footnotes
    assert SYNTHETIC_NOTE in t3r.footnotes[-1] and "transcribed" in t3r.footnotes[-1]
    sentinels = {"99.9 ± 9.9", "88.8 ± 8.8", "77.77 ± 7.77"}
    for tid, spec in out.items():
        if tid.endswith("_reported"):
            continue
        for frame in (spec.df, spec.extended):
            cells = set(frame.to_numpy().ravel()) if frame is not None else set()
            assert not cells & sentinels, tid
    assert render_markdown(t3r).startswith("### Table 3 (reported). ")


# --------------------------------------------------------------------------- review regressions


def _non_reported(specs: dict) -> dict:
    return {tid: spec for tid, spec in specs.items() if not tid.endswith("_reported")}


def test_missing_synthetic_flag_fails_safe(bundle, tmp_path):
    """A bundle that lost its synthetic flag but came from the mock backend is still shown as synthetic."""
    meta = copy.deepcopy(bundle.meta)
    meta.pop("synthetic")
    lost = AnalysisBundle(meta=meta, frames=bundle.frames)
    assert tables_mod.is_synthetic(lost)
    for tid, spec in build_tables(lost, reported=NO_REPORTED).items():
        assert spec.notes()[-1] == SYNTHETIC_NOTE and spec.synthetic, tid
    with pytest.raises(ValueError, match="SYNTHETIC"):
        write_tables(lost, tmp_path / "results" / "t", reported=NO_REPORTED)
    prov_only = copy.deepcopy(_real(bundle).meta)
    prov_only["provenance"] = {"backend": {"kind": "vllm", "synthetic": True}}
    assert tables_mod.is_synthetic(AnalysisBundle(meta=prov_only, frames={}))
    assert not tables_mod.is_synthetic(_real(bundle))


def test_far_notes_flag_partly_coupled_accepts_and_thin_seed_means(bundle, specs):
    summ = bundle.frame("policy_summary")
    # Consistent with the producers on the fixture: every partly coupled headline policy is named.
    for p in ("P1", "P2", "P3"):
        r = summ.set_index("policy").loc[p]
        k, n = int(r.n_accepted_gt_coupled), int(r.n_accepted)
        named = any(f"{p} {k} of {n}" in note for note in specs["T3"].footnotes)
        assert named == (0 < k < n), (p, k, n)
    forced = summ.copy()
    i = forced.index[forced["policy"] == "P2"][0]
    forced.loc[i, ["n_accepted", "n_accepted_gt_coupled", "n_seeds_far", "n_seeds"]] = [4, 2, 1, 2]
    out = build_tables(_with(bundle, policy_summary=forced), reported=NO_REPORTED)
    t3, t6, t9 = (out[t].footnotes for t in ("T3", "T6", "T9"))
    for notes in (t3, t6, t9):
        assert any(n.startswith("Partly zero by construction") and "P2 2 of 4" in n for n in notes)
    assert any("P2 1 of 2 seeds" in n for n in t3) and any("P2 1 of 2 seeds" in n for n in t9)
    assert not any("1 of 2 seeds" in n for n in t6)  # T6 is pooled, not a seed mean
    assert not out["T3"].df.iloc[1]["False-accept rate (%) ↓"].endswith("‡")  # partial: no dagger
    forced.loc[i, "n_accepted_gt_coupled"] = 4
    out = build_tables(_with(bundle, policy_summary=forced), reported=NO_REPORTED)
    assert out["T3"].df.iloc[1]["False-accept rate (%) ↓"].endswith("‡")
    partly = [n for n in out["T3"].footnotes if n.startswith("Partly zero by construction")]
    assert not any("P2 " in n for n in partly)  # fully coupled: dagger instead of the partial note


def test_partly_by_construction_cells_are_footnoted(bundle, specs):
    # Fixture: E1 greedy reruns at ages 2 and 4 are cache hits, so some T7b cells are partly by construction.
    summ = bundle.frame("pair_summary")
    partial = summ[(summ["by_construction_frac"] > 0) & (summ["by_construction_frac"] < 1)]
    assert len(partial), "fixture should contain partly by-construction cells"
    notes = " ".join(specs["T7b"].footnotes)
    for r in partial.itertuples(index=False):
        assert f"{r.env} at age {int(r.age)} {r.by_construction_frac * 100:.0f}%" in notes
    # T7: forcing a share on a measured row names that row; the cell itself carries no dagger.
    t7 = bundle.frame("table7").copy()
    m = (t7["age"] == 1) & (t7["env"] == "E1")
    t7.loc[m, "by_construction_frac"] = 0.25
    t7.loc[m, "by_construction"] = False
    out = build_tables(_with(bundle, table7=t7), reported=NO_REPORTED)["T7"]
    assert any(
        n.startswith("Partly identical by construction") and "age 1, Unchanged (E1) 25%" in n
        for n in out.footnotes
    )
    assert not out.df.iloc[2]["Drift inflation (pp)"].endswith("‡")
    # T4: a generation part that is zero by construction (E3 without physical reruns) is footnoted.
    ps = summ.copy()
    ps.loc[(ps["env"] == "E3") & (ps["age"] == PLAN.t4.age), "gen_by_construction_frac"] = 1.0
    t4 = build_tables(_with(bundle, pair_summary=ps), reported=NO_REPORTED)["T4"]
    assert any(
        n.startswith("Generation part zero by construction") and "Extraction change 100%" in n
        for n in t4.footnotes
    )
    assert not any(n.startswith("Generation part") for n in specs["T4"].footnotes)


def _coupled_counts(pairs: pd.DataFrame, ref: str) -> dict[tuple[str, int], tuple[int, int]]:
    """Independent re-derivation: (env, age) -> (accepts that cannot be false by construction, accepts)."""
    out = {}
    for (env, age), g in pairs.groupby(["env", "age"]):
        acc = g[g[f"dec_{ref}"].astype(bool)]
        ok = acc["gt_coupled"].astype(bool)
        if ref != "fresh":  # the reference prompt must be the current incumbent (rerun cell == GT cell)
            ok &= acc["ref_slot"] == acc["cur_slot"]
        if ref == "stored":  # and the stored generation must be that very rerun generation
            ok &= acc["by_construction"].astype(bool)
        out[(str(env), int(age))] = (int(ok.sum()), len(acc))
    return out


def test_partly_gt_coupled_far_cells_are_footnoted(bundle, specs):
    # Regression: a FAR cell where only SOME accepts are GT-coupled (e.g. rerun reference under greedy, when the
    # reference prompt is still the incumbent in some pairs) used to get neither a dagger nor a footnote.
    pairs = bundle.frame("pairs")
    t7b_notes = " ".join(specs["T7b"].footnotes)
    n_partial = 0
    for ref in ("stored", "rerun", "fresh"):
        for (env, age), (k, n) in _coupled_counts(pairs, ref).items():
            label = f"{env} at age {age}, {ref} reference {k} of {n}"
            if 0 < k < n:
                n_partial += 1
                assert label in t7b_notes, label
            else:
                assert label not in t7b_notes, label
    assert n_partial, "fixture should contain partly GT-coupled FAR cells"
    assert "Partly zero by construction" in t7b_notes
    # T7: stored-reference partials belong to both versions, rerun/fresh partials to the extended version only.
    t7 = specs["T7"]
    paper, ext = " ".join(t7.footnotes), " ".join(t7.notes(extended=True))
    ages = set(PLAN.t7.ages)
    for ref in ("rerun", "fresh"):
        for (env, age), (k, n) in _coupled_counts(pairs, ref).items():
            if 0 < k < n and age in ages and env in (PLAN.t7.unchanged_env, PLAN.t7.changed_env):
                status = "Unchanged" if env == PLAN.t7.unchanged_env else "Changed"
                label = f"age {age}, {status} ({env}), {ref} reference {k} of {n}"
                assert label in ext and label not in paper, label


def test_t8_by_construction_share_and_far_choice(bundle, specs):
    abl = bundle.frame("ablations")
    name = abl.loc[abl["n_pairs"] > 0, "ablation"].iloc[0]  # an ablation whose schedule changes within R
    row = abl.set_index("ablation").loc[name]
    pos = specs["T8"].df["Ablation"].str.startswith(f"{name}:").to_numpy().nonzero()[0][0]
    pairs = bundle.frame("pairs").copy()
    m = pairs["env"].isin(str(row.inflation_envs).split(",")) & (pairs["age"] == int(row.inflation_age))
    assert m.sum() == row.n_pairs
    assert not specs["T8"].df.iloc[pos]["Drift inflation (pp)"].endswith("‡")
    pairs.loc[m, "by_construction"] = True
    out = build_tables(_with(bundle, pairs=pairs), reported=NO_REPORTED)["T8"]
    assert out.df.iloc[pos]["Drift inflation (pp)"].endswith("‡") and BY_CONSTRUCTION_NOTE in out.footnotes
    idx = pairs.index[m]
    pairs.loc[idx[: len(idx) // 2], "by_construction"] = False
    share = pairs.loc[m, "by_construction"].mean()
    out = build_tables(_with(bundle, pairs=pairs), reported=NO_REPORTED)["T8"]
    assert not out.df.iloc[pos]["Drift inflation (pp)"].endswith("‡")
    assert any(f"{name} {share * 100:.0f}%" in n for n in out.footnotes)
    # Without the reference_age_variation column, A4 still reports its fixed-age policy's FAR.
    no_flag = bundle.frame("ablations").drop(columns=["reference_age_variation"])
    df = build_tables(_with(bundle, ablations=no_flag), reported=NO_REPORTED)["T8"].df
    assert df.iloc[3]["False-accept rate (%)"] == specs["T8"].df.iloc[3]["False-accept rate (%)"]
    assert df.iloc[0]["False-accept rate (%)"] == specs["T8"].df.iloc[0]["False-accept rate (%)"]
    # run_ablations records GT-coupled accepts, so the generic caveat is only for bundles without them.
    assert not any("do not record which accepts are coupled" in n for n in specs["T8"].footnotes)
    legacy = bundle.frame("ablations").drop(
        columns=[c for c in bundle.frame("ablations").columns if c.startswith("n_accepted_gt_coupled_")]
    )
    old = build_tables(_with(bundle, ablations=legacy), reported=NO_REPORTED)["T8"]
    assert any("do not record which accepts are coupled" in n for n in old.footnotes)


def test_t8_marks_gt_coupled_far(bundle):
    abl = bundle.frame("ablations").copy()
    abl["n_accepted_P1b"] = 4
    abl["n_accepted_gt_coupled_P1b"] = [4, 2, 0, 0]  # A1 all coupled, A2 partly, A3 none
    abl["n_accepted_P3"] = 3
    abl["n_accepted_gt_coupled_P3"] = 0  # A4 shows its fixed-age policy (P3 slot): not coupled
    out = build_tables(_with(bundle, ablations=abl), reported=NO_REPORTED)["T8"]
    far = out.df["False-accept rate (%)"].tolist()
    assert far[0].endswith("‡") and not any(f.endswith("‡") for f in far[1:])
    assert BY_CONSTRUCTION_NOTE in out.footnotes
    assert any(n.startswith("Partly zero by construction") and "A2 2 of 4" in n for n in out.footnotes)
    assert not any("A1 4 of 4" in n for n in out.footnotes)


def test_partial_run_default_plan_and_skipped_pairs_are_footnoted(bundle):
    partial = build_tables(
        _with(bundle, {"R_observed": {"0": 3, "1": 4}, "skipped_pairs": 7}), reported=NO_REPORTED
    )
    for tid, spec in _non_reported(partial).items():
        for ext in (False, True):
            notes = spec.notes(ext)
            prov = next(i for i, n in enumerate(notes) if n.startswith("Analysis plan"))
            assert any(n.startswith("Partial run: completed rounds (seed 0: 3)") for n in notes[:prov]), tid
    for tid in ("T4", "T7", "T7b", "T8", "T9"):
        assert any(
            n.startswith("7 candidate/reference pairs were skipped") for n in partial[tid].footnotes
        ), tid
    assert (
        partial["T3"]
        .footnotes[0]
        .startswith("Mean ± standard deviation over 2 seeds (20 test questions, 3 rounds")
    )
    no_plan = build_tables(_with(bundle, {"plan": {"t4": "not a section"}}), reported=NO_REPORTED)
    for tid, spec in no_plan.items():
        assert any("no valid analysis plan" in n for n in spec.footnotes), tid
    clean = build_tables(bundle, reported=NO_REPORTED)
    assert not any(
        "Partial run" in n or "no valid analysis plan" in n for s in clean.values() for n in s.footnotes
    )


def test_plan_variants_change_the_wording(bundle):
    plan = PLAN.model_dump(mode="json")
    plan["headline_policies"] = ["p1", "P2", "p3"]  # canonicalized, not looked up verbatim
    t3 = build_tables(_with(bundle, {"plan": plan}), reported=NO_REPORTED)["T3"].df
    assert t3["Policy"].tolist()[0] == "P1: Frozen reference" and DASH not in t3["Model calls ↓"].tolist()
    split = PLAN.model_dump(mode="json")
    split["gt"]["mode"] = "split_half"
    split["reference_mode"] = "chain"
    out = build_tables(_with(bundle, {"plan": split}), reported=NO_REPORTED)
    assert any(f"share of the {N // 2} test questions" in n for n in out["T4"].footnotes)
    assert any("(chain mode)" in n for n in out["T7"].footnotes)


def test_write_tables_removes_stale_managed_files(bundle, tmp_path):
    out = tmp_path / "tables"
    rep = {"T3": [{"policy": "P1", "gt_acc": "70.0 ± 1.0"}]}
    write_tables(bundle, out, reported=rep)
    assert (out / "T3_reported.md").exists() and (out / "T3_extended.tex").exists()
    (out / "notes.md").write_text("mine\n")
    (out / "T3_custom.md").write_text("mine\n")
    write_tables(bundle, out, formats=("md", "csv"), reported=NO_REPORTED, extended=False)
    assert not (out / "T3_reported.md").exists() and not (out / "T3_extended.md").exists()
    assert not (out / ALL_TABLES_EXTENDED).exists() and (out / ALL_TABLES).exists()
    assert (out / "T3_extended.tex").exists()  # a format that was not written is left alone
    assert (out / "notes.md").exists() and (out / "T3_custom.md").exists()  # not ours


def test_all_tables_header_lists_analysis_warnings(bundle, tmp_path):
    warned = _with(
        bundle, {"warnings": ["analysis plan aaa differs from the plan recorded at run time (bbb)"]}
    )
    write_tables(warned, tmp_path, formats=("md",), reported=NO_REPORTED)
    head = (tmp_path / ALL_TABLES).read_text(encoding="utf-8").split("### ")[0]
    assert "Analysis warnings (1)" in head and "differs from the plan recorded at run time" in head
    ext = build_tables(warned, reported=NO_REPORTED)["T1"].extended
    assert dict(zip(ext["Component"], ext["Configuration"], strict=True))["Analysis warnings"].startswith(
        "1: "
    )
