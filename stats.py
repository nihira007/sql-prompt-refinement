"""
stats.py
========

Small, dependency-free statistics used to judge whether a gain is real:

* ``wilson_interval``  95% confidence interval for an accuracy (k correct out of n)
* ``sign_test``        exact two-sided sign test on paired outcomes
* ``paired_counts``    how many questions a strategy fixed / broke versus a baseline
* ``mean_sd``          mean and sample standard deviation across repeated runs
"""

from __future__ import annotations

import math
from typing import Sequence


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion. Returns (low, high) as fractions in [0, 1]."""
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def sign_test(wins: int, losses: int) -> float:
    """
    Exact two-sided sign test: probability of a split at least as lopsided as
    ``wins`` vs ``losses`` if both strategies were equally good. Ties are ignored.
    """
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_counts(baseline: Sequence[bool], candidate: Sequence[bool]) -> tuple[int, int]:
    """(fixed, broken): questions the candidate gets right and the baseline wrong, and vice versa."""
    fixed = sum(c and not b for b, c in zip(baseline, candidate))
    broken = sum(b and not c for b, c in zip(baseline, candidate))
    return fixed, broken


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (0.0 for fewer than two values)."""
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def format_ci(successes: int, n: int) -> str:
    """'62% [31-86%]' style label."""
    lo, hi = wilson_interval(successes, n)
    pct = successes / n if n else 0.0
    return f"{pct:.0%} [{lo:.0%}-{hi:.0%}]"
