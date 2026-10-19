"""Tests for the concentration math in analyze.py.

`concentration_share` backs both the concentration figure and the printed
"worst 10% of nights = X% of deaths" headline, so it is worth pinning
independently of the (network-fetched) real dataset.
"""

import numpy as np
import pytest

from birddata import analyze


def test_perfectly_even_distribution():
    # ten identical nights: the worst 10% (one night) holds exactly 10%.
    nights = [5] * 10
    assert analyze.concentration_share(nights, 0.10) == pytest.approx(0.10)


def test_all_deaths_on_one_night():
    # one catastrophic night dominates; the worst 10% captures all of it.
    nights = [100] + [0] * 9
    assert analyze.concentration_share(nights, 0.10) == pytest.approx(1.0)


def test_order_independent():
    a = analyze.concentration_share([1, 50, 3, 2, 4], 0.20)
    b = analyze.concentration_share([50, 4, 3, 2, 1], 0.20)
    assert a == b == pytest.approx(50 / 60)


def test_at_least_one_night_counts():
    # frac so small it would floor to zero nights still counts the worst one.
    nights = [10, 1, 1, 1, 1]
    assert analyze.concentration_share(nights, 0.001) == pytest.approx(10 / 14)


def test_full_fraction_is_everything():
    nights = [3, 7, 5]
    assert analyze.concentration_share(nights, 1.0) == pytest.approx(1.0)


def test_empty_and_zero_are_safe():
    assert analyze.concentration_share([], 0.10) == 0.0
    assert analyze.concentration_share([0, 0, 0], 0.10) == 0.0


def test_matches_naive_computation():
    rng = np.random.default_rng(0)
    nights = rng.integers(0, 30, size=200)
    frac = 0.10
    ordered = np.sort(nights)[::-1]
    k = max(1, int(len(ordered) * frac))
    expected = ordered[:k].sum() / ordered.sum()
    assert analyze.concentration_share(nights, frac) == pytest.approx(expected)


# --- correlation_stats -------------------------------------------------------


def test_correlation_r_matches_numpy():
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    y = 0.5 * x + rng.normal(size=500)
    st = analyze.correlation_stats(x, y)
    assert st["r"] == pytest.approx(np.corrcoef(x, y)[0, 1])
    assert st["n"] == 500


def test_correlation_ci_brackets_r_and_is_ordered():
    rng = np.random.default_rng(2)
    x = rng.normal(size=400)
    y = 0.4 * x + rng.normal(size=400)
    st = analyze.correlation_stats(x, y)
    assert st["ci_low"] < st["r"] < st["ci_high"]


def test_correlation_strong_is_significant():
    rng = np.random.default_rng(3)
    x = rng.normal(size=300)
    y = 2 * x + rng.normal(scale=0.1, size=300)  # near-perfect
    st = analyze.correlation_stats(x, y)
    assert st["r"] > 0.9
    assert st["p"] < 1e-6


def test_correlation_none_is_not_significant():
    rng = np.random.default_rng(4)
    x = rng.normal(size=1000)
    y = rng.normal(size=1000)  # independent
    st = analyze.correlation_stats(x, y)
    assert st["p"] > 0.05


def test_correlation_drops_nan_pairs():
    x = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    y = np.array([2.0, 4.0, 6.0, 8.0, 100.0])
    st = analyze.correlation_stats(x, y)
    assert st["n"] == 4
    assert st["r"] == pytest.approx(1.0)


def test_correlation_degenerate_inputs_are_nan():
    # too few points, and a constant series: both yield NaN, no crash
    assert np.isnan(analyze.correlation_stats([1, 2], [3, 4])["r"])
    assert np.isnan(analyze.correlation_stats([5, 5, 5, 5], [1, 2, 3, 4])["p"])


def test_fmt_p():
    assert analyze._fmt_p(0.0001) == "p < 0.001"
    assert analyze._fmt_p(0.042) == "p = 0.042"
    assert analyze._fmt_p(float("nan")) == "p = n/a"
