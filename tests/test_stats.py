import pytest

from tai_bot.stats import mcnemar_exact, paired_bootstrap, pct_ci, wilson


def test_wilson_matches_reference_values() -> None:
    lo, hi = wilson(37, 40)  # reference: 0.801 - 0.974
    assert lo == pytest.approx(0.801, abs=0.002)
    assert hi == pytest.approx(0.974, abs=0.002)
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0
    assert wilson(10, 10)[0] < 1.0  # a perfect score still has uncertainty


def test_pct_ci_format() -> None:
    assert pct_ci(37, 40) == "92% (37/40; CI 80-97%)"
    assert pct_ci(0, 0) == "n/a"


def test_mcnemar_counts_only_disagreements() -> None:
    a = [True] * 10 + [True] * 6 + [False] * 4
    b = [True] * 10 + [False] * 6 + [False] * 4
    only_a, only_b, p = mcnemar_exact(a, b)
    assert (only_a, only_b) == (6, 0)
    assert p == pytest.approx(2 / 2**6)
    assert mcnemar_exact([True, False], [True, False])[2] == 1.0


def test_paired_bootstrap_interval_contains_the_mean() -> None:
    a = [5, 4, 5, 4, 5, 5, 4, 5]
    b = [2, 3, 2, 3, 2, 2, 3, 2]
    mean, lo, hi = paired_bootstrap(a, b)
    assert mean == pytest.approx(2.25)
    assert lo <= mean <= hi and lo > 0
