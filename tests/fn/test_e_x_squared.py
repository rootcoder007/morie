"""Tests for morie.fn.e_x_squared: values recomputed from first principles."""

import math

from morie.fn.e_x_squared import e_x_squared


def test_second_moment_from_a_pmf():
    vals, probs = [1.0, 2.0, 6.0], [0.2, 0.5, 0.3]
    mu = sum(v * p for v, p in zip(vals, probs))
    ex2 = sum(v * v * p for v, p in zip(vals, probs))
    sigma = math.sqrt(ex2 - mu * mu)
    assert abs(e_x_squared(sigma, mu)["e_x2"] - ex2) < 1e-13
