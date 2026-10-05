# DriftLab architecture (binding implementation spec)

This document is the contract between modules. When code and this document disagree, fix one of them
in the same change. Research-facing definitions are repeated in `docs/METHODS.md` (written for the paper).

## 0. Conventions
* Python ≥ 3.10, src layout (`src/driftlab`), `from __future__ import annotations` everywhere.
* Lint: `ruff check . && ruff format --check .` (config in pyproject). Tests: `pytest -m "not slow"`.
* Works with **pandas 2.2 and 3.x** (pandas 3 = copy-on-write + `str` dtype by default: never rely on
  chained assignment, never compare dtype to `object` for strings) and numpy ≥ 1.26 / 2.x.
* Never create directories named `lib/ env/ build/ dist/ downloads/ var/ parts/` (ignored by .gitignore).
* Torch / transformers / vllm are **optional** imports inside their backend modules only. Core, analysis,
  reporting and the app must import without them.
* Determinism: no wall-clock randomness. All randomness flows from `driftlab.keys` seeds.
* Synthetic data (mock backend) is always marked `runs.synthetic = 1`; this flag can never be cleared.

## 1. Vocabulary
| Term | Meaning |
|---|---|
| seed | one repeated pilot run (0, 1, 2); controls proposer sampling, error sampling and sampling seeds |
| round t | optimization round 1..R (R = 11). Round 0 = initial state |
| slot k | prompt p_k. Slot 0 = initial prompt; slot t = candidate proposed at round t |
| trajectory | the shared chain of slots of one seed, built on the DEV split only (Phase A) |
| inc_slot[t] | trajectory incumbent when candidate t is evaluated (`inc_slot[0]=inc_slot[1]=0`) |
| decoding | `greedy` (T=0) or `t02` (T=0.2); every parameter explicit (top_p=1, top_k=0, rep=1.0, 640 tokens) |
| extractor | `v1` strict / `v2` lenient (frozen; source-hashed) |
| environment | decoding × extractor (+ model, engine, max_new_tokens): E1 greedy+v1, E2 t02+v1, E3 greedy+v2, E4 t02+v2 |
| cell | logical generation event `(seed, split, decoding, slot, draw_kind, draw, item)` → `gen_key` |
| draw | `("round", r)` = generation event at round r; `("gt", g)` = independent ground-truth draw; `("audit", a)` |
| generation | one physical model output, content-addressed by `gen_key` (`driftlab.keys.gen_key`) |
| physical | a cell whose generation was forced fresh (greedy with nonce) or is an independent sample |
| reference | stored outputs + **scores at storage time** of some slot, used as a comparator |
| age | rounds since the reference outputs were generated (t − created_round) |

## 2. Module map and ownership
```
src/driftlab/
  config.py            ExperimentConfig, AnalysisPlan, load_config(path, overrides), load_plan(path)   [DONE]
  environments.py      Decoding, Environment, EnvSchedule, diff(), build_environments()                [DONE]
  keys.py              gen_key(), sample_seed(), proposer_seed(), rng_seed(), engine_fingerprint()        [DONE]
  data.py              Item, load_dev/load_eval -> DevSplit|EvalSplit, canonical_gold(answer) -> str (items.gold)
                       (extraction.parse_gold(answer) -> Fraction is the numeric twin used for scoring)
  extraction/          common.py, v1.py, v2.py, __init__ (REGISTRY, extract(name, text), tags, is_correct)
  backends/            base.py [DONE], mock.py, hf.py, gumbel.py, vllm_backend.py, openai_compat.py,
                       __init__.make_backend(cfg, answer_key=None)
  store/               schema.sql [DONE], store.py (Store), shards.py (ShardWriter, replay)
  engine.py            GenerationEngine (cache-aware, chunked, ledgered, time-budgeted)
  prompts/             initial_v1.txt, meta_v1.txt, fallback_edits.yaml (package data)
  proposer.py          build_meta_prompt(), parse_candidate(), validate_candidate(), fallback_candidate()
  trajectory.py        run_trajectory(...)  (Phase A)
  planning.py          plan_matrix(...) -> list[CellSpec]; requests for Phase B
  scoring.py           score_all(store, run_id): every registered extractor over every unscored generation
  audit.py             run_audit(...)  determinism audit
  pipeline.py          Pipeline(cfg, run_dir, backend=None).run(stages, max_minutes) / status()
  analysis/
    cube.py            Cube, Trajectory, empty_cube, set_cell, make_trajectory   [DONE: contract]
    loader.py          load_cube(db, run_id, split), load_trajectories(db, run_id)
    metrics.py         paired counts, decide(), wilson(), mcnemar_exact(), mean_sd()
    bootstrap.py       paired item bootstrap utilities
    allpairs.py        all_pairs(...) -> PairResult; summarize_pairs(...)
    policies.py        PolicySpec, CostConvention, simulate(...), simulate_all(...)
    ablations.py       run_ablations(...), randomize_schedules(...)
    hypotheses.py      evaluate_hypotheses(...)
    bundle.py          analyze(run_dir) -> AnalysisBundle (DataFrames, persisted under exports/analysis/)
  reporting/tables.py  build_tables(bundle) -> dict["T1".."T10", TableSpec]; render md/csv/tex
  estimate.py          estimate(cfg, gpu, backend, ...) -> Estimate
  provenance.py        collect_provenance(cfg, backend) -> dict
  colab.py             ColabSession (Drive restore / snapshot / budgeted run)
  cli.py               `driftlab` entry point
app/                   Streamlit dashboard (Home.py + pages/), reads run dirs read-only
```

## 3. Phase A — trajectory (`trajectory.py`, `proposer.py`)
* Inputs: config, `GenerationEngine`, `Store`, `DevSplit` (type guard: the eval split is never loaded or
  passed in Phase A; `DevError` asserts `split == "train"`), seeds.
* Round 0: slot 0 = initial prompt (`prompts/initial_v1.txt`, no few-shot examples); dev run under the
  trajectory env (E1: greedy + v1) → dev correctness of slot 0. Dev cells: `split='train'`,
  `decoding_id='greedy'`, `slot=k`, `draw=('round', k)`.
* Round t = 1..R (all seeds in lockstep so each round is one big batch):
  1. `inc = inc_slot[t]`; errors = dev items where slot `inc` is incorrect under v1.
  2. Sample `n_errors` errors with `random.Random(rng_seed(seed, "errors", t))`; build the meta-prompt from
     `prompts/meta_v1.txt` with the incumbent text, and for each error: question, the response truncated to
     head 300 + " … " + tail 300 chars, gold answer.
  3. Proposer generation: decoding `cfg.proposer_decoding()` (T=0.7), seed `proposer_seed(seed, t, attempt)`;
     parse the LAST `<prompt>…</prompt>`; validate: length in [min_chars, max_chars]; contains `\boxed` when
     `require_format_instruction`; normalized text differs from every earlier slot of this seed; no ≥3-digit
     number copied from the shown gold answers; no nested tags. First valid attempt wins.
  4. Else fallback: incumbent text + one seeded sentence from `fallback_edits.yaml` not already present
     (`origin='fallback'`, `is_fallback=1`).
  5. Candidate dev run (E1); advance per `advance_rule` (`dev_gt`: cand > inc; `dev_ge`; `always`).
     `mode: static` never advances (all candidates proposed from slot 0).
  6. Commit the `trajectory_rounds` row LAST (it marks the round complete; resume skips completed rounds).
* Ledger purposes: `trajectory_dev` (incumbent/slot-0 dev runs), `candidate_dev`, `proposer`.

## 4. Phase B — measurement matrix (`planning.py`, `scoring.py`, `audit.py`)
Cells are ALWAYS planned for the full triangle so that the cube is uniform; the *mode* only decides
which cells are physically distinct generations:

* For every seed s, decoding d ∈ {greedy, t02}, slot k, round r ∈ [k, R]: cell `(s,'test',d,k,('round',r))`.
* **greedy**: request seed = None. Nonce = None (cache hit on the creation generation) except where the cell
  is physical:
  * `physical_greedy_reruns: all` → every r > k gets `nonce = f"rerun:s{seed}:{r}"` (the seed is part of the
    nonce so slot 0 — the same prompt in every seed — gets an independent physical rerun per seed).
  * `ages` → cell (k, r) is physical if some pair (i, j=r) with `inc_slot[i] == k` (reference_mode
    incumbent) or `i == k` (chain) has `j − i ∈ physical_ages`. (Plan both reference modes' needs.)
  * `none` → no nonces. Physical cells in `ages` mode use the same nonce format as `all`.
  * Creation cells (r = k) are never given a nonce and are `physical=1`.
* **t02 full**: request seed = `sample_seed(s, "eval", "t02", k, "round", r, "test", n)` → every round is an
  independent draw (`physical=1`).
* **t02 lean**: seed uses `draw=-1` for every r → all rounds of a slot share one sample (cache hits,
  `physical=0` except r = k).
* GT draws: for sampling decodings only, cells `(s,'test','t02',k,('gt',g))` for g < gt_draws with
  seed `sample_seed(s, "gt", "t02", k, "gt", g, "test", n)`.
* Audit (`audit.enabled`): for `audit.seed`, slots {0, last trajectory incumbent}, each decoding,
  `repeats` repeats: draws `('audit', a)` with nonce `f"audit:s{seed}:{a}"` for BOTH decodings; t02 audit
  requests reuse the creation cell's sampling seed, so (with the nonce forcing a fresh generation) they test
  per-request seed reproducibility — without the nonce they would be cache hits, identical by construction.
  Requests are shuffled and chunked differently from the main matrix. Results → `audit_results` (pct_text_identical, pct_correct_flip vs the creation cell).
* Execution order: group by draw (each round's requests together); chunk size from config per backend.
* Scoring: `score_all` runs every registered extractor on every generation that lacks a score row for the
  extractor's CURRENT source hash and writes `scores` (insert; an existing row is replaced only when its
  `ext_hash` differs). No model calls. Store inserts are "insert-or-ignore on key conflicts only"
  (`ON CONFLICT DO NOTHING`), so CHECK/NOT NULL violations still raise.
* Ledger purposes: `eval_matrix`, `gt_draw`, `audit`.

## 4b. Pipeline (`pipeline.py`) — the orchestration contract used by the CLI, Colab and tests
```python
STAGES = ("data", "trajectory", "matrix", "score", "audit", "analyze")
class Pipeline:
    def __init__(self, cfg: ExperimentConfig, run_dir: str | Path, backend: Backend | None = None, *,
                 run_id: str | None = None,             # default cfg.run.name
                 shard_dir: str | Path | None = None,  # default <run_dir>/shards (Colab: a Drive folder)
                 allow_engine_change: bool = False, allow_config_change: bool = False,
                 checkpoint_hook: Callable[[str], None] | None = None,  # called after each committed stage/round
                 log: Callable[[str], None] = print): ...
    def open(self) -> None            # store, backend (make_backend(cfg, answer_key) if None), provenance, ensure_run
    def run(self, stages: Sequence[str] | None = None, max_minutes: float | None = None) -> dict
    def status(self) -> dict          # per stage: planned / done / complete flag (derived from the data, not flags)
    def close(self) -> None
```
* Run dir layout: `config.yaml` (resolved config), `plan.yaml` (copy of the analysis plan), `store.sqlite`,
  `shards/`, `exports/analysis/` (bundle), `exports/tables/` (T1–T10 md/csv/tex), `provenance.json`.
* Every stage is idempotent and resumable: it computes planned − done from the store and does only the rest.
  `max_minutes` is a soft deadline checked between chunks/rounds; on expiry `run()` returns
  `{"status": "budget_exhausted", ...}` after committing all finished work (re-running continues).
* Re-opening a run dir with a different config hash raises unless `allow_config_change`.
* The analyze stage calls `driftlab.analysis.bundle.analyze(run_dir)` and then
  `driftlab.reporting.tables.write_tables(bundle, run_dir / "exports" / "tables")`.

## 5. Generation engine (`engine.py`)
```python
class GenerationEngine:
    def __init__(self, backend, store, run_id, *, chunk_size=512, shard_writer=None,
                 deadline: float | None = None, fail_after_chunks: int | None = None, ...): ...
    def run(self, tasks: list[CellTask], *, stage: str, purpose: str, seed: int | None = None,
            round_: int | None = None) -> list[GenRecord]          # aligned with tasks
```
* `CellTask(cell: CellKey | None, request: GenRequest, physical: bool = False)`. `CellKey` = (run_id, seed, split, decoding_id,
  slot, draw_kind, draw, item_idx). `cell=None` for proposer calls (still stored in `generations`).
* Steps: compute `rendered = backend.render(system, user)`; `gen_key(...)` with the backend's engine
  fingerprint; dedupe; look up existing generations; generate the missing ones in deterministic chunk order;
  per chunk ONE transaction writes generations + cells + a ledger row (requested/executed/cache-hit counts,
  tokens, wall time) and, if configured, one immutable shard file. Cells whose generation was a cache hit
  are still written (they are logical events).
* Between chunks: raise `BudgetExhausted` if `deadline` has passed; `DRIFTLAB_FAIL_AFTER_CHUNKS` (env var)
  or `fail_after_chunks` raises `InjectedFailure` (resume tests).
* Physical vs logical counts are both kept: logical = requested, executed = cache misses.
* Cells are first-write-wins: re-planning an existing cell with a different gen_key (nonce, seed, prompt,
  decoding or engine changed) raises `CellConflict` — a changed matrix plan needs a new run dir.
* Sampling requests MUST carry an explicit seed from `driftlab.keys` (an unseeded sample would be cached and
  reused by every draw); the engine enforces this.
* Backend results with `finish_reason="error"` are paid-for calls: they get an extra ledger row with stage
  `"<stage>:error"` (include in executed cost; exclude when reconciling ledger counts with stored generations).
* Call `run()` once per seed (pass `seed=`) so the ledger can be split per seed.
* The engine refuses to write into a run whose stored `engine_fp` differs (`EngineMismatch`), and a synthetic
  backend makes the run's `synthetic` flag sticky.
* Restore order (Colab): `replay_shards(store, dir)` BEFORE `ShardWriter.for_store(...)` (else `ShardLogAhead`);
  a missing shard raises `ShardGap` unless `allow_gaps=True`.

## 6. Analyses (all post-hoc over the cube; no model calls)
### 6.1 Notation
`V(s,d,k,draw,x)` = cube.vec; `GT(s,d,k,r,x)` = cube.gt_acc (greedy → round-r cell; t02 → mean of gt draws).
Environment c = (d_c, x_c). Storage env st = plan.storage_env (E1 → (greedy, v1)).

### 6.2 All-pairs drift analysis (`allpairs.py`) — source of T4, T7, T8 (factorial, unconfounded)
For every seed s, env c ∈ {E1..E4}, candidate j ∈ 1..R, creation round i ∈ 0..j (age = j − i):
* `ref = inc_slot[i]` (incumbent mode; `i` in chain mode)
* `stored   = V(s, d_st, ref, ('round', i), x_st)`  (scores kept from storage time)
* `rescored = V(s, d_st, ref, ('round', i), x_c)`   (same text, current extractor)
* `rerun    = V(s, d_c,  ref, ('round', j), x_c)`   (same prompt regenerated under c at round j)
* `fresh    = V(s, d_c, inc_slot[j], ('round', j), x_c)` (current incumbent, current env)
* `cand     = V(s, d_c, j, ('round', j), x_c)`
* For each reference R∈{stored, rescored, rerun, fresh}: `w_R = Σ(cand ∧ ¬R)`, `l_R = Σ(¬cand ∧ R)`, ties = N−w−l.
* `win_rate_R = w_R/N`; `inflation = (w_stored − w_rerun)/N`; `infl_extract = (w_stored − w_rescored)/N`;
  `infl_generation = (w_rescored − w_rerun)/N` (sums to inflation).
* Decisions `dec_R = decide(w_R, l_R, N, rule)`; `flip = dec_stored ≠ dec_rerun`.
* GT (under c): `gt_cand = GT(j)`, `gt_cur = GT(inc_slot[j])`, `gt_ref = GT(ref)` at round j.
  `fa_cur_R = dec_R ∧ gt_cand ≤ gt_cur` (primary), `fa_ref_R = dec_R ∧ gt_cand ≤ gt_ref`.
* `inflation := infl_extract + infl_generation` (equal to `(w_stored − w_rerun)/N` up to 1 ulp; makes the
  decomposition exact; a zero stays exactly 0.0).
* `by_construction = (same_gen_frac(stored_cell, rerun_cell) == 1) ∧ (x_st == x_c)` → "‡ identical by construction".
  `gen_by_construction`: the generation part is 0 because rerun and stored are the same generation (‡ on that
  part). `gt_coupled`: under greedy (or `same_draw`) GT reuses the decision cells, so FAR against a reference that
  IS the GT cell is 0 by construction (‡). Byte-identical text from a physical rerun is a *measured* reproduction
  (`Cube.same_text_frac`), not a by-construction zero.
* `split_half` GT mode: decisions use items [0, N/2), GT uses items [N/2, N).
* Keep per-item arrays `(cand∧¬stored) − (cand∧¬rerun)` for the paired bootstrap.
* `summarize_pairs(df, by=("env","age"))`: pooled means, FAR = Σfa/Σdec with Wilson 95% CI, flip rate,
  n_pairs, n_accepts, per-seed means → mean ± sd; inflation 95% CI from a paired item bootstrap that uses ONE
  resampled item index vector per replicate for all arrays, stratified by seed (B from plan).

### 6.3 Policy simulation (`policies.py`) — source of T3, T5, T6, T9
Policies: `P1` frozen (never refresh, never adopt: reference = round-0 output of slot 0 forever — matches the
teammate footnote), `P1b` frozen + adopt-on-promote, `P2` per-batch refresh, `P3` env-triggered refresh,
`P4_k<k>` age-triggered (age ≥ k), `P5` component-aware (regenerate if decoding/model/engine changed;
re-score stored text at zero cost if only the extractor changed), `ORACLE` (decides on GT; cost N per round),
`FIXEDAGE_k<k>` (ablation A4: reference = incumbent output stored k rounds earlier under that round's env).
All except P1, ORACLE and FIXEDAGE adopt the candidate's already-computed outputs as the new reference on
promotion (0 calls). FIXEDAGE: `inc_hist[r]` = shadow incumbent when candidate r is evaluated (`inc_hist[0]=0`);
the reference at round t is `inc_hist[max(t−k,0)]`'s output at that round under that round's env (FIXEDAGE_k0 ≡ P2);
it is charged N only when the reference snapshot actually changes. `RoundDecision.gt_coupled` marks rounds whose
reference is the shadow incumbent's own GT cell (decision ≡ GT comparison); `policy_summary` reports
`n_gt_coupled` / `n_accepted_gt_coupled`, and a FAR whose accepts are all coupled is rendered with ‡.
`split_half` requires disjoint `items` (decision) and `gt_items` (GT).
```
round 0: inc = 0; ref = Ref(slot 0, round 0, env=sched[0], scores=V(s,d0,0,('round',0),x0), source='initial')
for t in 1..R:
    env_t = sched[t] = (d, x); cand = V(s, d, t, ('round', t), x)
    maybe refresh (policy rule) -> ref = Ref(inc, t, env_t, V(s, d, inc, ('round', t), x), 'refresh')  [+N calls]
    w, l = wins/losses(cand, ref.scores); accepted = decide(w, l, N, rule)   (ORACLE: GT(t) > GT(inc))
    false_accept = accepted and GT(t) <= GT(inc)   (both under env_t, GT spec)
    log RoundDecision(..., ref_age = t - ref.round)
    if accepted: inc = t; if adopt: ref = Ref(t, t, env_t, cand, 'adopt')   [0 calls]
final: GT accuracy of inc under plan.gt.canonical_env and under sched[R]
```
**Cost conventions** (`CostConvention`):
* `teammate_v1`: per round D (incumbent dev) + 1 (proposer) + N (candidate eval); + N per refresh;
  no initial-reference cost; rescores free; ORACLE + N per round. With R=11, N=D=200 and a 2-change schedule
  this reproduces **P1 4411, P2 6611, P3 4811** (P5 4611). A unit test pins these numbers.
* `full`: also counts candidate dev runs (D per round), every proposer attempt, the initial reference (N).
* Executed calls (cache misses actually paid in this run) come from the `ledger` table and are reported
  separately from the policies' logical costs. Candidate-generation calls are reported separately from
  reference/evaluation calls.

### 6.4 Ablations, schedule randomization, hypotheses
* A1–A4 from the plan: re-run `simulate` with each schedule (A4 uses FIXEDAGE_k1) and the all-pairs envs that
  the schedule visits. Columns: drift inflation (age 3, over the schedule's changed envs), FAR (P1, P1b, P3),
  GT acc and calls (P3).
* Schedule randomization (exploratory): n random schedules with exactly n_changes change rounds in 1..R and
  segment envs drawn from {E2,E3,E4} (each differs from the previous); distribution of FAR and calls per policy.
* Hypotheses: H1 inflation@age3 for E2–E4 vs E1 control with paired-bootstrap CIs; H2 FAR differences
  (P1−P2, P1b−P2) with bootstrap over items (resimulate per replicate); H3 reference-call ratio P3/P2 and
  FAR(P3) − FAR(P2) vs the equivalence margin. Report effect, CI, n; status ∈ {supported, not supported,
  inconclusive}. Never change hypotheses after looking.

### 6.5 Analysis bundle (`analysis/bundle.py`)
`analyze(run_dir) -> AnalysisBundle` loads cube + trajectories + plan, runs everything, and persists
DataFrames as CSV under `runs/<run>/exports/analysis/` plus a `bundle.json` index (plan hash, config hash,
synthetic flag, created_at). `load_bundle(run_dir)` reads them back (used by the app and reporting).
Required frames: `pairs` (one row per s, env, j, i), `pair_summary` (by env × age), `policy_rounds`
(one row per s, policy, t), `policy_summary` (per policy: mean±sd + pooled), `policy_refs` (reference
snapshots), `candidates` (T5 rows), `ablations`, `schedule_random`, `hypotheses`, `ledger_summary`,
`audit`, `truncation`, `trajectory` (per s, t).

## 7. Reporting (`reporting/tables.py`)
`build_tables(bundle) -> dict[str, TableSpec]` with `TableSpec(id, title, df, footnotes: list[str])`;
`render_markdown / render_csv / render_latex`. Every footnote list ends with the plan hash, config hash and
(if applicable) `SYNTHETIC DATA — not experimental results`. Column specs:
* T1 configuration key/value; T2 E1–E4 (decoding, extractor, reference version, current env, changed
  components, purpose); T3 policies (GT acc canonical mean±sd, GT acc final-env, FAR mean±sd, FAR pooled k/n
  [Wilson], calls: candidate-gen / eval / reference / total, refreshes, rescores, executed calls);
  T4 env × (stored win, rerun win, inflation [CI], extraction part, generation part, flip rate, n_pairs, ‡);
  T5 per candidate (id `s{seed}-r{t}`, seed, round, env@t, incumbent slot/acc, candidate acc, Δ, outcome,
  dev Δ, advanced, P1/P2/P3/Oracle decisions); T6 per policy (evaluated, accepted, accepted w/o GT
  improvement, FAR); **T7** age {0,1,3,5,10} × {unchanged E1, changed E4}: stored win, rerun win, inflation
  [CI], FAR_cur(stored), FAR_cur(rerun), FAR_cur(fresh), flip rate, n_pairs, n_accepts (+ T7b full grid);
  T8 A1–A4; T9 per seed + mean±sd row; T10 dashboard components (page, what it shows, data source, status).
* `driftlab tables` refuses to write a synthetic run into `results/` (only into the run's exports).
* ‡ marks values that are identical/zero BY CONSTRUCTION (by_construction, gen_by_construction, gt_coupled);
  every table that can contain one carries the explanatory footnote.
* The teammate's reported numbers (`results/reported/teammate.yaml`) are shown side-by-side, never merged.

## 8. CLI (`driftlab`)
`estimate | run | status | analyze | tables | demo | extract | audit | freeze-plan | verify-extractors |
dashboard | info`. `run -c CONFIG --run-dir DIR [--stages ...] [--max-minutes M] [--set k=v]` is idempotent.
`demo` = run `configs/demo_mock.yaml` + analyze + tables into `runs/demo_mock`.

## 8b. Backend notes
* openai_compat: `base_url` is NOT part of the engine fingerprint (Colab tunnel URLs change); the user-declared
  `backend.openai_compat.server_tag` describes the server engine instead. Main runs against `vllm serve` must use
  `--generation-config vllm --no-enable-prefix-caching` (prefix caching would bias physical-rerun drift to zero).
* HF CPU tests use `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5@ce8d0bf270b28c8fab026ced69fe8aa14b0f0eda`.

## 9. Dashboard (`app/`)
Streamlit multi-page app; sidebar run picker over `runs/*/` (and `DRIFTLAB_RUN`); read-only DB; red
"SYNTHETIC DATA" banner when `synthetic=1`; pages: Home, 1 Environment Tracker, 2 Reference Manager,
3 Reference Rerun (+ live rerun ≤ 20 items via mock/openai_compat, logged to `interactive_events` only),
4 Candidate Comparison, 5 Ground Truth, 6 Drift Dashboard (T7 first), 7 Refresh Policies, 8 Round Replay,
9 Reproducibility, 10 Extractor Lab, 11 Paper Tables.
