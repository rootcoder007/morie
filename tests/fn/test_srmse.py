"""Tests for morie.fn.srmse: recompute from the definition."""

import math

from morie.fn.srmse import root_mean_squared_error

X = [2.5, -1.0, 4.25, 0.5, 3.0, -2.75, 1.5, 6.0]


def test_rmse():
    Y = [v + 0.1 * i for i, v in enumerate(X)]
    r = root_mean_squared_error(X, Y)
    assert abs(r.value - math.sqrt(math.fsum((0.1 * i) ** 2 for i in range(8)) / 8)) < 1e-15
    assert abs(r.value**2 - r.extra["mse"]) < 1e-15
