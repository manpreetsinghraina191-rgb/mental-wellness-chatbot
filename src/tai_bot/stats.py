"""Small-sample statistics for the evaluation scripts (no scipy needed).

- wilson(k, n):          95% interval for a proportion; well-behaved at small n and near 0/100%
                         (the plain +-1.96*SE interval is not).
- mcnemar_exact(a, b):   paired pass/fail comparison on the SAME items: only items where the
                         two systems disagree carry information; exact two-sided binomial p.
- paired_bootstrap(a, b): 95% interval for mean(a - b) over the same items (judge scores).
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

Z95 = 1.959964


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return (lo, hi)


def pct_ci(k: int, n: int) -> str:
    """'92% (37/40; 95% CI 80-97%)'"""
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.0f}% ({k}/{n}; CI {100 * lo:.0f}-{100 * hi:.0f}%)"


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> tuple[int, int, float]:
    """Returns (only_a, only_b, p): items where only system a passed, only b passed, and the
    exact two-sided p-value for 'both systems pass equally often'."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    only_a = sum(x and not y for x, y in zip(a, b, strict=True))
    only_b = sum(y and not x for x, y in zip(a, b, strict=True))
    n = only_a + only_b
    if n == 0:
        return (0, 0, 1.0)
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return (only_a, only_b, min(1.0, 2 * tail))


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], iters: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of (a - b) with a 95% percentile bootstrap interval."""
    if len(a) != len(b) or not a:
        raise ValueError("need two non-empty paired samples of equal length")
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(iters))
    return (sum(diffs) / n, means[int(0.025 * iters)], means[int(0.975 * iters) - 1])
