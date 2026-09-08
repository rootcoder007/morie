"""Tests for sieg.siegel_repeated (Siegel 1982 repeated medians)."""

import pytest

from morie.fn.sieg import siegel_repeated


def test_exact_line_is_recovered_exactly():
    r = siegel_repeated(list(range(1, 8)), [3 * v - 2 for v in range(1, 8)])
    assert r["slope"] == 3.0
    assert r["intercept"] == -2.0
    assert max(abs(v) for v in r["residuals"]) == 0.0


def test_hand_computed_five_point_fit():
    """Inner medians by hand: 1.625, 1.5, 1.25, 2, 1.125 -> median 1.5;
    levels -0.5, 0, -2.5, 1, -1.5 -> median -0.5."""
    r = siegel_repeated([1, 2, 3, 4, 5], [1, 3, 2, 7, 6])
    assert r["slope"] == 1.5
    assert r["intercept"] == -0.5


def test_breakdown_three_of_nine_corrupted():
    x = list(range(1, 10))
    y = [2.0 * v for v in x]
    for i in (1, 4, 7):
        y[i] = 1e6
    r = siegel_repeated(x, y)
    assert r["slope"] == 2.0
    assert r["intercept"] == 0.0
    assert r["breakdown_point"] == 4 / 9


def test_past_the_breakdown_point_it_does_break():
    x = list(range(1, 10))
    y = [2.0 * v for v in x]
    for i in range(5):
        y[i] = -1e6
    assert abs(siegel_repeated(x, y)["slope"]) > 1000


def test_regression_equivariance():
    x = [1, 2, 3, 4, 5]
    y = [1, 3, 2, 7, 6]
    a, b, c = -2.5, 0.75, 4.0
    base = siegel_repeated(x, y)["slope"]
    shifted = siegel_repeated(x, [a * y[i] + b * x[i] + c for i in range(5)])
    assert abs(shifted["slope"] - (a * base + b)) < 1e-12


def test_error_paths():
    with pytest.raises(ValueError):
        siegel_repeated([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        siegel_repeated([1.0], [1.0])
    with pytest.raises(ValueError):
        siegel_repeated([2.0, 2.0, 2.0], [1.0, 5.0, 9.0])
