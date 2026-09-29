"""Tests for morie.fn.exponential_waiting_density: values recomputed from first principles."""

import math

from morie.fn.exponential_waiting_density import exponential_waiting_density


def test_density_and_normalisation():
    lam = 0.7
    assert abs(exponential_waiting_density(1.3, lam)["density"] - lam * math.exp(-lam * 1.3)) < 1e-16
    h = 1e-3
    mass = sum(exponential_waiting_density((i + 0.5) * h, lam)["density"] for i in range(40000)) * h
    assert abs(mass - (1 - math.exp(-lam * 40))) < 1e-6
