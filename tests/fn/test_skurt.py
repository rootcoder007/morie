"""Tests for morie.fn.skurt: recompute from the definition."""

import math

from morie.fn.skurt import kurtosis_coeff

X = [2.5, -1.0, 4.25, 0.5, 3.0, -2.75, 1.5, 6.0]


def test_population_excess_kurtosis():
    m = sum(X) / 8
    m2 = math.fsum((v - m) ** 2 for v in X) / 8
    m4 = math.fsum((v - m) ** 4 for v in X) / 8
    assert abs(kurtosis_coeff(X).value - (m4 / m2**2 - 3)) < 1e-14
    # two-point symmetric distribution: kurtosis 1, excess -2
    assert abs(kurtosis_coeff([-1.0, 1.0]).value + 2.0) < 1e-15
