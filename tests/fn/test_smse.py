"""Tests for morie.fn.smse: recompute from the definition."""

import math

from morie.fn.smse import mean_squared_error

X = [2.5, -1.0, 4.25, 0.5, 3.0, -2.75, 1.5, 6.0]


def test_mse():
    Y = [v + 0.1 * i for i, v in enumerate(X)]
    assert abs(mean_squared_error(X, Y).value - math.fsum((0.1 * i) ** 2 for i in range(8)) / 8) < 1e-15
