"""Tests for morie.fn.sskew: recompute from the definition."""

import math

from morie.fn.sskew import skewness_coeff

X = [2.5, -1.0, 4.25, 0.5, 3.0, -2.75, 1.5, 6.0]


def test_population_skewness():
    m = sum(X) / 8
    m2 = math.fsum((v - m) ** 2 for v in X) / 8
    m3 = math.fsum((v - m) ** 3 for v in X) / 8
    assert abs(skewness_coeff(X).value - m3 / m2**1.5) < 1e-14
    assert skewness_coeff([2.0, 2.0]).value == 0.0
    # symmetric data have zero skewness
    assert abs(skewness_coeff([1.0, 2.0, 3.0, 4.0]).value) < 1e-15
