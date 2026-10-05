"""Pre-registered hypotheses H1-H3 (docs/ARCHITECTURE.md section 6.4; ``plan.primary_contrasts``).

* **H1** (unit ``pp``): drift inflation at reference age ``plan.t4.age`` under each changed environment
  (E2, E3, E4) minus the unchanged control ``plan.t7.unchanged_env`` (E1) at the same age. The CI is a paired
  item bootstrap with ONE shared index vector per replicate for every array, seed and environment
  (``bootstrap_indices(n_decision_items, B, seed)``, the same replicates :func:`summarize_pairs` uses, so the
  raw-inflation rows reproduce the T4 intervals). Differences use the (seed, j, i) pairs present under both
  environments. Per env: ``effect detected`` iff the difference CI excludes 0 (two-sided: the sign under a
  decoding change is not pre-registered). Overall: ``supported`` iff some changed env shows an effect,
  ``not supported`` if none does, ``inconclusive`` if no changed env has any pair. Notes carry ‡ whenever some
  (``k of n``) or all pairs of an env are identical by construction, or their generation part is.
* **H2** (unit ``pp of FAR``): ``FAR(P1) - FAR(P2)`` and ``FAR(P1b) - FAR(P2)``, FAR pooled over the seeds
  BOTH policies of the contrast cover (false accepts / accepts). The CI resamples items (one index vector per
  replicate shared by every seed and policy; ``split_half`` resamples the decision half and the GT half
  separately) and re-simulates the policies with :func:`driftlab.analysis.policies.simulate`; replicates where
  a FAR is undefined (no accepts) are dropped and counted in the note. Status, in order: ``inconclusive`` if
  either policy has fewer than :data:`MIN_ACCEPTS` accepts in total, if every accept of both policies is
  GT-coupled (both FARs are 0 by construction, ‡), or if there is no CI; ``supported`` if the CI excludes 0
  (two-sided, as pre-registered; the note states the direction); otherwise ``not supported``.
* **H3**: the reference-call ratio ``calls_reference(P3) / calls_reference(P2)`` (deterministic, unit
  ``ratio``) and ``FAR(P3) - FAR(P2)`` (bootstrap CI and the accept / GT-coupling / no-CI guards as in H2)
  against ``plan.h3_equivalence_margin_pp``: ``supported`` iff ratio < 1 and the CI lies inside
  (-margin, +margin); ``not supported`` iff ratio >= 1 or the CI lies entirely outside the margin; else
  ``inconclusive``. The note counts GT-coupled accepts (``RoundDecision.gt_coupled``: decision == GT comparison
  by construction, so such an accept cannot be false).

Everything is deterministic given ``plan.bootstrap.seed`` (or ``seed``).
"""

from __future__ import annotations

import dataclasses
import functools
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from driftlab.analysis.allpairs import PairResult
from driftlab.analysis.bootstrap import (
    bootstrap_diff_ci,
    bootstrap_indices,
    paired_bootstrap_ci,
    percentile_interval,
)
from driftlab.analysis.cube import Cube, MissingCell, Trajectory
from driftlab.analysis.policies import (
    PolicyRun,
    as_environments,
    as_trajectories,
    parse_policy,
    simulate,
    simulate_all,
)
from driftlab.environments import Environment, EnvSchedule
from driftlab.keys import rng_seed

HYPOTHESIS_COLUMNS: tuple[str, ...] = (
    "hypothesis",
    "contrast",
    "subject",
    "estimate",
    "ci_lo",
    "ci_hi",
    "unit",
    "n",
    "status",
    "note",
)
MIN_ACCEPTS = 5  # H2: fewer accepts (pooled over seeds) for either policy -> inconclusive
H2_CONTRASTS: tuple[tuple[str, str], ...] = (("P1", "P2"), ("P1b", "P2"))
H3_CONTRAST: tuple[str, str] = ("P3", "P2")
SUPPORTED, NOT_SUPPORTED, INCONCLUSIVE = "supported", "not supported", "inconclusive"
EFFECT, NO_EFFECT, NO_DATA, DESCRIPTIVE = "effect detected", "no effect detected", "no data", "descriptive"

_NAN = float("nan")


def _row(
    hypothesis: str,
    contrast: str,
    subject: str,
    estimate: float,
    lo: float,
    hi: float,
    unit: str,
    n: int,
    status: str,
    note: str = "",
) -> dict:
    return {
        "hypothesis": hypothesis,
        "contrast": contrast,
        "subject": subject,
        "estimate": float(estimate),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "unit": unit,
        "n": int(n),
        "status": status,
        "note": note,
    }


def _excludes_zero(lo: float, hi: float) -> bool:
    return bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))


def _join(*parts: str) -> str:
    return "; ".join(p for p in parts if p)


# --------------------------------------------------------------------------- H1


def _expected_pairs(trajs: Mapping[int, Trajectory], age: int) -> int:
    """Pairs (s, j, i = j - age), 1 <= j <= R_s, i >= 0, that a complete matrix would give one env."""
    return int(sum(max(0, t.R - max(1, age) + 1) for t in trajs.values()))


def _pair_keys(df: pd.DataFrame, env: str, age: int) -> list[tuple[int, int, int]]:
    m = (df["env"].astype(str) == env) & (df["age"].astype(int) == age)
    sub = df.loc[m]
    return sorted((int(s), int(j), int(i)) for s, j, i in zip(sub["seed"], sub["j"], sub["i"], strict=True))


def h1_rows(
    pr: PairResult,
    trajs: Mapping[int, Trajectory],
    plan,
    envs: Mapping[str, object],
    *,
    B: int,
    seed: int,
) -> list[dict]:
    """H1 rows: raw inflation per env (``descriptive``), changed-minus-control per env, and an overall row."""
    age = int(plan.t4.age)
    control = str(plan.t7.unchanged_env)
    df = pr.df
    env_ids = [str(e) for e in envs]
    changed = [e for e in env_ids if e != control]
    n_dec = int(len(pr.item_index))
    idx = bootstrap_indices(n_dec, B, seed) if n_dec and B > 0 else np.zeros((0, n_dec), dtype=np.int64)
    expected = _expected_pairs(trajs, age)
    at = df.loc[df["age"].astype(int) == age] if not df.empty else df
    keys = {e: _pair_keys(df, e, age) for e in env_ids} if not df.empty else {e: [] for e in env_ids}
    skipped = {
        e: sum(1 for sk in pr.skipped if sk[1] == e and sk[2] is not None and sk[2] - sk[3] == age)
        for e in env_ids
    }

    def arrays(env: str, ks: Sequence[tuple[int, int, int]]) -> list[np.ndarray]:
        return [pr.item_diffs[(s, env, j, i)] for s, j, i in ks if (s, env, j, i) in pr.item_diffs]

    def flags(env: str) -> str:
        """‡ notes for pairs of ``env`` at ``age`` that are (partly) zero by construction, with counts."""
        sub = at.loc[at["env"].astype(str) == env] if len(at) else at
        n = int(len(sub))
        if not n:
            return ""
        # by_construction implies gen_by_construction, so gbc >= bc
        bc = int(sub["by_construction"].astype(bool).sum())
        gbc = int(sub["gen_by_construction"].astype(bool).sum())
        parts = []
        if bc == n:
            parts.append(
                f"‡ {env} at age {age} is identical by construction (rerun = stored generation): 0 by design"
            )
        elif bc:
            parts.append(
                f"‡ {bc} of {n} {env} pairs at age {age} are identical by construction (rerun = stored "
                "generation): their inflation is 0 by design"
            )
        if gbc > bc:
            share = "" if gbc == n else f" in {gbc} of {n} pairs"
            parts.append(f"‡ generation part of {env} is 0 by construction{share} (cache-hit reruns)")
        return _join(*parts)

    def missing(env: str) -> str:
        have = len(keys[env])
        if expected and have < expected:
            sk = f" ({skipped[env]} skipped for missing cells)" if skipped[env] else ""
            return f"{expected - have} of {expected} expected pairs missing{sk}"
        return ""

    rows: list[dict] = []
    for env in [control, *changed]:
        if env not in keys:
            continue
        arr = arrays(env, keys[env])
        est, lo, hi = paired_bootstrap_ci(arr, B, seed, indices=idx) if arr else (_NAN,) * 3
        note = _join("control" if env == control else "", flags(env), missing(env))
        rows.append(
            _row(
                "H1",
                f"inflation@age{age}",
                env,
                est * 100,
                lo * 100,
                hi * 100,
                "pp",
                len(arr),
                DESCRIPTIVE,
                note,
            )
        )

    statuses: list[str] = []
    ctrl_keys = set(keys.get(control, []))
    for env in changed:
        common = sorted(set(keys[env]) & ctrl_keys)
        a, b = arrays(env, common), arrays(control, common)
        unmatched = len(keys[env]) + len(ctrl_keys) - 2 * len(common)
        no_ci = ""
        if a and b:
            est, lo, hi = bootstrap_diff_ci(a, b, B, seed, indices=idx)
            if np.isfinite(lo) and np.isfinite(hi):
                status = EFFECT if _excludes_zero(lo, hi) else NO_EFFECT
            else:  # B = 0: a point estimate alone cannot show (or rule out) an effect
                status, no_ci = NO_DATA, "no bootstrap CI"
        else:
            est = lo = hi = _NAN
            status = NO_DATA
        statuses.append(status)
        note = _join(
            no_ci,
            flags(control),
            flags(env),
            missing(env),
            f"{unmatched} pair(s) without a counterpart in {env}/{control} dropped" if unmatched else "",
        )
        rows.append(
            _row(
                "H1",
                f"inflation@age{age}({env}) - inflation@age{age}({control})",
                env,
                est * 100,
                lo * 100,
                hi * 100,
                "pp",
                len(common),
                status,
                note,
            )
        )
    if any(s == EFFECT for s in statuses):
        overall, note = SUPPORTED, ""
    elif any(s == NO_EFFECT for s in statuses):
        overall, note = NOT_SUPPORTED, ""
    else:
        overall, note = INCONCLUSIVE, f"no decidable contrast at age {age} under any changed environment"
    detected = [e for e, s in zip(changed, statuses, strict=True) if s == EFFECT]
    no_data = [e for e, s in zip(changed, statuses, strict=True) if s == NO_DATA]
    note = _join(
        note,
        f"effect detected under {', '.join(detected)}" if detected else "",
        f"no data under {', '.join(no_data)}" if no_data and len(no_data) < len(changed) else "",
        f"B={B}",
    )
    rows.append(
        _row(
            "H1",
            f"inflation@age{age}: any of {','.join(changed)} vs {control}",
            "overall",
            _NAN,
            _NAN,
            _NAN,
            "pp",
            sum(s in (EFFECT, NO_EFFECT) for s in statuses),
            overall,
            note,
        )
    )
    return rows


# --------------------------------------------------------------------------- policy bootstrap (H2, H3)


@dataclass
class _PolicyStats:
    """Pooled point counts and per-replicate counts of one policy over ``seeds``.

    ``by_seed`` holds the same statistics per seed (empty for hand-built stats), so a contrast can be restricted
    to the seeds both of its policies cover (:meth:`restrict`).
    """

    seeds: list[int]
    n_accepted: int
    n_false: int
    n_coupled: int
    ref_calls_mean: float
    rep_acc: np.ndarray | None = None  # (B,) accepts per replicate (None: bootstrap unavailable)
    rep_fa: np.ndarray | None = None
    by_seed: dict[int, _PolicyStats] = field(default_factory=dict)

    @property
    def far(self) -> float:
        return self.n_false / self.n_accepted if self.n_accepted else _NAN

    @property
    def all_coupled(self) -> bool:
        """Every accept is GT-coupled (decision == GT comparison), so this FAR is 0 by construction."""
        return self.n_accepted > 0 and self.n_coupled == self.n_accepted

    def restrict(self, seeds: Iterable[int]) -> _PolicyStats:
        """The statistics pooled over ``seeds`` only (``self`` when nothing is dropped or no per-seed data)."""
        keep = {int(s) for s in seeds}
        if set(self.seeds) <= keep or not self.by_seed:
            return self
        return _pool([self.by_seed[s] for s in self.seeds if s in keep and s in self.by_seed])


def _pool(parts: Sequence[_PolicyStats]) -> _PolicyStats:
    """Pool per-seed statistics (sums of counts and of replicate counts; mean reference calls over seeds)."""
    reps = bool(parts) and all(p.rep_acc is not None and p.rep_fa is not None for p in parts)
    return _PolicyStats(
        seeds=[s for p in parts for s in p.seeds],
        n_accepted=int(sum(p.n_accepted for p in parts)),
        n_false=int(sum(p.n_false for p in parts)),
        n_coupled=int(sum(p.n_coupled for p in parts)),
        ref_calls_mean=float(np.mean([p.ref_calls_mean for p in parts])) if parts else _NAN,
        rep_acc=np.sum([p.rep_acc for p in parts], axis=0) if reps else None,
        rep_fa=np.sum([p.rep_fa for p in parts], axis=0) if reps else None,
        by_seed={p.seeds[0]: p for p in parts if len(p.seeds) == 1},
    )


class _MemoEnv(Environment):
    """An :class:`Environment` whose fingerprint is memoised.

    The fingerprint is a pure function of the frozen fields and the extractor tags; without the memo the
    per-replicate re-simulation spends about half its time re-hashing the same few environments.
    """

    def fingerprint(self, extractor_tags: Mapping[str, str] | None = None) -> str:
        key = (
            None
            if extractor_tags is None
            else tuple(sorted((str(k), str(v)) for k, v in extractor_tags.items()))
        )
        return _memo_fingerprint(self, key)


@functools.lru_cache(maxsize=4096)
def _memo_fingerprint(env: Environment, tags_key: tuple[tuple[str, str], ...] | None) -> str:
    return Environment.fingerprint(env, None if tags_key is None else dict(tags_key))


def _memo_envs(envmap: Mapping[str, Environment]) -> dict[str, Environment]:
    """``envmap`` with plain Environments replaced by equal-valued :class:`_MemoEnv` copies."""
    out: dict[str, Environment] = {}
    for k, e in envmap.items():
        if type(e) is Environment:
            e = _MemoEnv(**{f.name: getattr(e, f.name) for f in dataclasses.fields(e)})
        out[k] = e
    return out


def _matches_plan(run: PolicyRun, sched: EnvSchedule, gt_mode: str) -> bool:
    if str(run.gt_mode) != str(gt_mode):
        return False
    if not run.schedule:
        return True
    R = max((d.round for d in run.rounds), default=0)
    return EnvSchedule(dict(run.schedule)).as_list(R) == sched.as_list(R)


def _point_runs(
    runs: Sequence[PolicyRun],
    names: Sequence[str],
    cube: Cube,
    trajs: Mapping[int, Trajectory],
    plan,
    envs,
    n_dev: int,
    tags: Mapping[str, str] | None,
) -> tuple[dict[str, list[PolicyRun]], str]:
    """Runs per policy under the plan schedule (first per seed) and an error note; a missing policy is
    simulated here (a policy that cannot be simulated stays empty, so H1 rows are never lost)."""
    sched = plan.env_schedule()
    out: dict[str, dict[int, PolicyRun]] = {n: {} for n in names}
    for r in runs:
        if r.policy.name in out and _matches_plan(r, sched, plan.gt.mode):
            out[r.policy.name].setdefault(int(r.seed), r)
    todo = [n for n in names if not out[n]]
    # only seeds whose trajectory the cube covers (simulate raises MissingCell otherwise)
    cube_seeds = {int(s) for s in cube.seeds}
    ok = {s: t for s, t in trajs.items() if s in cube_seeds and cube.n_slots > t.R}
    errors: list[str] = []
    for name in todo if ok else []:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                extra = simulate_all(
                    cube, ok, plan, envs, n_dev, extractor_tags=tags, policies=[name], on_missing="skip"
                )
        except (MissingCell, KeyError, IndexError, ValueError) as e:  # e.g. an extractor with no score rows
            errors.append(f"policy {name} not simulated: {type(e).__name__}: {e}")
            continue
        if not extra:  # every seed skipped (missing cells, an extractor without score rows, ...)
            why = str(caught[0].message) if caught else "no simulable seed"
            errors.append(f"policy {name} not simulated: {why}")
            continue
        for r in extra:
            out[r.policy.name].setdefault(int(r.seed), r)
    return {n: [v[s] for s in sorted(v)] for n, v in out.items()}, "; ".join(errors)


def _policy_stats(
    point: Mapping[str, list[PolicyRun]],
    cube: Cube,
    trajs: Mapping[int, Trajectory],
    plan,
    envs,
    n_dev: int,
    tags: Mapping[str, str] | None,
    B_policy: int,
    seed: int,
) -> tuple[dict[str, _PolicyStats], str]:
    """Point stats and the shared item-resampling replicates of every policy (per seed and pooled); returns
    (stats, error note)."""
    per: dict[str, dict[int, _PolicyStats]] = {
        name: {
            int(r.seed): _PolicyStats(
                seeds=[int(r.seed)],
                n_accepted=int(r.n_accepted),
                n_false=int(r.n_false_accepts),
                n_coupled=int(r.n_accepted_gt_coupled),
                ref_calls_mean=float(r.reference_calls),
            )
            for r in rs
        }
        for name, rs in point.items()
    }

    def pooled() -> dict[str, _PolicyStats]:
        return {name: _pool([d[s] for s in sorted(d)]) for name, d in per.items()}

    N = int(cube.n_items)
    if B_policy <= 0 or N == 0:
        return pooled(), "no policy bootstrap (B=0)" if B_policy <= 0 else "no items"
    if plan.gt.mode == "split_half":
        h = N // 2
        dec = bootstrap_indices(h, B_policy, rng_seed("hypotheses", "policy-bootstrap", "decision", seed))
        gt = bootstrap_indices(N - h, B_policy, rng_seed("hypotheses", "policy-bootstrap", "gt", seed)) + h
    else:
        dec = bootstrap_indices(N, B_policy, rng_seed("hypotheses", "policy-bootstrap", seed))
        gt = dec
    sched = plan.env_schedule()
    menvs = _memo_envs(as_environments(envs))
    errors: list[str] = []
    for name, by_seed in per.items():
        if not by_seed:
            continue
        spec = parse_policy(name)
        reps: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        try:
            for s in sorted(by_seed):
                traj = trajs[s]
                acc = np.zeros(B_policy, dtype=np.int64)
                fa = np.zeros(B_policy, dtype=np.int64)
                for b in range(B_policy):
                    run = simulate(
                        cube,
                        traj,
                        sched,
                        menvs,
                        spec,
                        plan.promotion_rule,
                        gt_mode=plan.gt.mode,
                        canonical_env=plan.gt.canonical_env,
                        conv=plan.cost_convention,
                        n_dev=n_dev,
                        extractor_tags=tags,
                        items=dec[b],
                        gt_items=gt[b],
                    )
                    acc[b] = run.n_accepted
                    fa[b] = run.n_false_accepts
                reps[s] = (acc, fa)
        except (MissingCell, KeyError, IndexError, ValueError) as e:  # this policy's CI stays NaN
            errors.append(f"policy bootstrap failed for {name}: {type(e).__name__}: {e}")
            continue
        for s, (acc, fa) in reps.items():
            by_seed[s].rep_acc, by_seed[s].rep_fa = acc, fa
    return pooled(), "; ".join(errors)


def _paired(stats: Mapping[str, _PolicyStats], a: str, b: str) -> tuple[_PolicyStats, _PolicyStats, str]:
    """Both policies of a contrast restricted to the seeds they both cover (a FAR contrast must be paired)."""
    sa, sb = stats[a], stats[b]
    if set(sa.seeds) == set(sb.seeds):
        return sa, sb, ""
    common = sorted(set(sa.seeds) & set(sb.seeds))
    note = (
        f"paired over the seeds both policies cover: {common} "
        f"({a} simulated on {sorted(sa.seeds)}, {b} on {sorted(sb.seeds)})"
    )
    return sa.restrict(common), sb.restrict(common), note


def _far_diff(a: _PolicyStats, b: _PolicyStats) -> tuple[float, float, float, int, int]:
    """(estimate pp, lo pp, hi pp, n valid replicates, n dropped replicates) of FAR(a) - FAR(b)."""
    est = (a.far - b.far) * 100.0
    if a.rep_acc is None or b.rep_acc is None or a.rep_fa is None or b.rep_fa is None:
        return est, _NAN, _NAN, 0, 0
    ok = (a.rep_acc > 0) & (b.rep_acc > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (a.rep_fa / a.rep_acc - b.rep_fa / b.rep_acc) * 100.0
    lo, hi = percentile_interval(d[ok])
    return est, lo, hi, int(ok.sum()), int((~ok).sum())


def _accepts_note(pairs: Sequence[tuple[str, _PolicyStats]]) -> str:
    parts = []
    for n, st in pairs:
        dag = " ‡ all GT-coupled (FAR 0 by construction)" if st.all_coupled else ""
        parts.append(f"{n}: {st.n_false}/{st.n_accepted} false accepts, {st.n_coupled} GT-coupled{dag}")
    return "; ".join(parts)


def _far_guard(a: str, b: str, sa: _PolicyStats, sb: _PolicyStats, lo: float, hi: float) -> str:
    """Why a FAR contrast cannot be judged (``""`` if it can): too few accepts, a difference that is 0 by
    construction (every accept of both policies GT-coupled), or no bootstrap CI."""
    if sa.n_accepted < MIN_ACCEPTS or sb.n_accepted < MIN_ACCEPTS:
        return f"fewer than {MIN_ACCEPTS} accepts for {a} or {b}"
    if sa.all_coupled and sb.all_coupled:
        return (
            f"‡ every accept of {a} and {b} is GT-coupled: both FARs are 0 by construction, so their difference "
            "is not a measurement"
        )
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "no bootstrap CI"
    return ""


def h2_rows(stats: Mapping[str, _PolicyStats], B_policy: int, err: str) -> list[dict]:
    rows = []
    for a, b in H2_CONTRASTS:
        sa, sb, paired_note = _paired(stats, a, b)
        est, lo, hi, n_ok, n_drop = _far_diff(sa, sb)
        why = _far_guard(a, b, sa, sb, lo, hi)
        if why:
            status = INCONCLUSIVE
        elif _excludes_zero(lo, hi):
            status = SUPPORTED
            why = (
                f"CI excludes 0: FAR({a}) > FAR({b})"
                if lo > 0
                else f"CI excludes 0 with FAR({a}) < FAR({b}): {a} admitted FEWER false accepts than {b}"
            )
        else:
            status, why = NOT_SUPPORTED, "CI includes 0"
        note = _join(
            why,
            paired_note,
            _accepts_note(((a, sa), (b, sb))),
            f"{n_drop} of {B_policy} replicates dropped (FAR undefined)" if n_drop else "",
            f"B={B_policy}",
            err,
        )
        rows.append(
            _row(
                "H2",
                f"FAR({a}) - FAR({b})",
                f"{a}-{b}",
                est,
                lo,
                hi,
                "pp of FAR",
                sa.n_accepted + sb.n_accepted,
                status,
                note,
            )
        )
    return rows


def h3_rows(stats: Mapping[str, _PolicyStats], plan, B_policy: int, err: str) -> list[dict]:
    a, b = H3_CONTRAST
    sa, sb, paired_note = _paired(stats, a, b)
    margin = float(plan.h3_equivalence_margin_pp)
    ratio = (
        sa.ref_calls_mean / sb.ref_calls_mean
        if sb.ref_calls_mean and np.isfinite(sb.ref_calls_mean)
        else _NAN
    )
    ratio_status = INCONCLUSIVE if not np.isfinite(ratio) else (SUPPORTED if ratio < 1 else NOT_SUPPORTED)
    est, lo, hi, n_ok, n_drop = _far_diff(sa, sb)
    why = _far_guard(a, b, sa, sb, lo, hi)
    if why:
        far_status = INCONCLUSIVE
    elif lo > -margin and hi < margin:
        far_status = SUPPORTED
    elif lo >= margin or hi <= -margin:
        far_status = NOT_SUPPORTED
    else:
        far_status = INCONCLUSIVE
    if ratio_status == SUPPORTED and far_status == SUPPORTED:
        overall = SUPPORTED
    elif ratio_status == NOT_SUPPORTED or far_status == NOT_SUPPORTED:
        overall = NOT_SUPPORTED
    else:
        overall = INCONCLUSIVE
    far_note = _join(
        f"equivalence margin ±{margin:g} pp",
        why,
        paired_note,
        _accepts_note(((a, sa), (b, sb))),
        f"{n_drop} of {B_policy} replicates dropped (FAR undefined)" if n_drop else "",
        f"B={B_policy}",
        err,
    )
    n_seeds = len(set(sa.seeds) | set(sb.seeds))
    return [
        _row(
            "H3",
            f"calls_reference({a}) / calls_reference({b})",
            f"{a}/{b}",
            ratio,
            _NAN,
            _NAN,
            "ratio",
            n_seeds,
            ratio_status,
            _join(
                f"mean reference calls per seed: {a} {sa.ref_calls_mean:g}, {b} {sb.ref_calls_mean:g}",
                paired_note,
            ),
        ),
        _row(
            "H3",
            f"FAR({a}) - FAR({b})",
            f"{a}-{b}",
            est,
            lo,
            hi,
            "pp of FAR",
            sa.n_accepted + sb.n_accepted,
            far_status,
            far_note,
        ),
        _row(
            "H3",
            f"ratio < 1 and |FAR({a}) - FAR({b})| CI within ±{margin:g} pp",
            "overall",
            _NAN,
            _NAN,
            _NAN,
            "",
            n_seeds,
            overall,
            _join(f"cost: {ratio_status}; FAR equivalence: {far_status}", "‡" if why.startswith("‡") else ""),
        ),
    ]


# --------------------------------------------------------------------------- entry point


def evaluate_hypotheses(
    pr: PairResult,
    runs: Sequence[PolicyRun],
    cube: Cube,
    trajs: Mapping[int, Trajectory] | Sequence[Trajectory],
    plan,
    envs: Mapping[str, object] | None,
    *,
    n_dev: int,
    extractor_tags: Mapping[str, str] | None = None,
    B: int | None = None,
    seed: int | None = None,
    B_policy: int = 500,
) -> pd.DataFrame:
    """H1-H3 as one frame with columns :data:`HYPOTHESIS_COLUMNS` (see the module docstring).

    ``runs`` are the headline policy runs (plan schedule); policies needed but absent (P1, P1b, P2, P3) are
    simulated with the plan's settings. ``B`` (default ``plan.bootstrap.B``) replicates drive the H1 item
    bootstrap, ``B_policy`` the H2/H3 re-simulation bootstrap; ``seed`` defaults to ``plan.bootstrap.seed``.
    """
    B = int(plan.bootstrap.B if B is None else B)
    seed = int(plan.bootstrap.seed if seed is None else seed)
    envmap = as_environments(envs)
    tmap = as_trajectories(trajs)
    rows = h1_rows(pr, tmap, plan, envmap, B=B, seed=seed)
    names = list(dict.fromkeys(n for pair in (*H2_CONTRASTS, H3_CONTRAST) for n in pair))
    point, sim_err = _point_runs(runs, names, cube, tmap, plan, envmap, n_dev, extractor_tags)
    stats, err = _policy_stats(point, cube, tmap, plan, envmap, n_dev, extractor_tags, int(B_policy), seed)
    err = _join(sim_err, err)
    rows += h2_rows(stats, int(B_policy), err)
    rows += h3_rows(stats, plan, int(B_policy), err)
    return pd.DataFrame(rows, columns=list(HYPOTHESIS_COLUMNS))


__all__ = [
    "H2_CONTRASTS",
    "H3_CONTRAST",
    "HYPOTHESIS_COLUMNS",
    "MIN_ACCEPTS",
    "evaluate_hypotheses",
    "h1_rows",
    "h2_rows",
    "h3_rows",
]
