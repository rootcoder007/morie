"""Tests for morie.fn.density_expectation: recompute Morin (2016) from the formula."""

import math

from morie.fn.density_expectation import density_expectation


def test_trapezoid_expectation():
    xs = [i / 10 for i in range(21)]
    rho = [3 * x * x / 8 for x in xs]  # density on [0, 2]
    f = [x * r for x, r in zip(xs, rho)]
    ref = math.fsum((f[i] + f[i + 1]) * (xs[i + 1] - xs[i]) / 2 for i in range(20))
    assert abs(density_expectation(xs, rho)["expectation"] - ref) < 1e-12
    # the exact mean is 3/2; the trapezoid rule is within h^2
    assert abs(ref - 1.5) < 0.01
