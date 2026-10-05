"""Paired-comparison metrics, the promotion rule, and interval estimates."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Paired:
    """Paired outcome counts of a candidate against a comparator on the same items."""

    wins: int  # candidate correct, comparator incorrect
    losses: int  # candidate incorrect, comparator correct
    both: int  # both correct
    neither: int  # both incorrect

    @property
    def n(self) -> int:
        return self.wins + self.losses + self.both + self.neither

    @property
    def ties(self) -> int:
        return self.both + self.neither

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else float("nan")

    @property
    def loss_rate(self) -> float:
        return self.losses / self.n if self.n else float("nan")

    @property
    def tie_rate(self) -> float:
        return self.ties / self.n if self.n else float("nan")


def paired(cand: np.ndarray, ref: np.ndarray) -> Paired:
    """2x2 paired counts. Inputs are boolean (or 0/1) vectors over the same items."""
    c = np.asarray(cand, dtype=bool)
    r = np.asarray(ref, dtype=bool)
    if c.shape != r.shape:
        raise ValueError(f"shape mismatch {c.shape} vs {r.shape}")
    return Paired(
        wins=int((c & ~r).sum()),
        losses=int((~c & r).sum()),
        both=int((c & r).sum()),
        neither=int((~c & ~r).sum()),
    )


def binom_sf_half(k: int, n: int) -> float:
    """P(X >= k) for X ~ Binomial(n, 0.5), computed exactly."""
    if n <= 0:
        return 1.0
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    total = sum(math.comb(n, i) for i in range(k, n + 1))
    return total / (2**n)


def mcnemar_exact_one_sided(wins: int, losses: int) -> float:
    """One-sided exact McNemar p-value for H1: candidate better (wins > losses)."""
    return binom_sf_half(wins, wins + losses)


def decide(wins: int, losses: int, n: int, rule) -> bool:
    """Promotion decision of a reference-based policy.

    ``rule`` is a :class:`driftlab.config.PromotionRule` (or any object with ``kind``, ``margin``, ``tau``,
    ``alpha`` attributes):

    * ``net_win``  accept iff (wins - losses) / n > margin
    * ``win_rate`` accept iff wins / n >= tau and wins > losses
    * ``mcnemar``  accept iff wins > losses and the one-sided exact McNemar p-value < alpha
    """
    if n <= 0:
        return False
    kind = getattr(rule, "kind", "net_win")
    if kind == "net_win":
        return (wins - losses) / n > float(getattr(rule, "margin", 0.0))
    if kind == "win_rate":
        return wins / n >= float(getattr(rule, "tau", 0.05)) and wins > losses
    if kind == "mcnemar":
        return wins > losses and mcnemar_exact_one_sided(wins, losses) < float(getattr(rule, "alpha", 0.10))
    raise ValueError(f"unknown promotion rule {kind!r}")


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    """Proportion with Wilson score interval: (p, lo, hi). NaNs when n == 0."""
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def mean_sd(values: Iterable[float], ddof: int = 1) -> tuple[float, float, int]:
    """Mean and sample sd over finite values (sd = 0 when only one value). Returns (mean, sd, n)."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), 0)
    sd = float(arr.std(ddof=ddof)) if arr.size > ddof else 0.0
    return (float(arr.mean()), sd, int(arr.size))


def fmt_mean_sd(values: Sequence[float], scale: float = 100.0, digits: int = 1) -> str:
    m, s, n = mean_sd(values)
    if n == 0:
        return "—"
    return f"{m * scale:.{digits}f} ± {s * scale:.{digits}f}"


def safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")
