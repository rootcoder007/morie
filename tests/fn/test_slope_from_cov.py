"""Tests for morie.fn.slope_from_cov: values recomputed from first principles."""

from morie.fn.slope_from_cov import slope_from_cov


def test_least_squares_slope():
    x = [1.0, 2.0, 3.0, 5.0, 8.0]
    y = [2.0, 2.5, 4.0, 4.5, 9.0]
    mx, my = sum(x) / 5, sum(y) / 5
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sum((a - mx) ** 2 for a in x)
    assert abs(slope_from_cov(x, y)["slope"] - b) < 1e-14
