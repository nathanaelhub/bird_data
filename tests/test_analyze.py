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
