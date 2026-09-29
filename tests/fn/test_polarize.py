"""Tests for polarize (polarization and party-system measures)."""

import math

import pytest

from morie.fn.polarize import (
    earth_movers_distance,
    esteban_ray_index,
    party_system_indices,
    polarization_indices,
    polarization_trend,
)


def test_polarization_indices_recomputed():
    x = [-1.0, -0.8, 0.9, 1.2, 0.1, -0.5]
    n = 6
    m = sum(x) / n
    d = [v - m for v in x]
    var = sum(t * t for t in d) / (n - 1)
    gmd = sum(abs(a - b) for a in x for b in x) / n**2
    m2 = sum(t * t for t in d) / n
    g1 = sum(t**3 for t in d) / n / m2**1.5
    g2 = sum(t**4 for t in d) / n / m2**2 - 3
    G1 = g1 * math.sqrt(n * (n - 1)) / (n - 2)
    G2 = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6)
    bc = (G1 * G1 + 1) / (G2 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    r = polarization_indices(x)
    assert r["variance"] == pytest.approx(var, rel=1e-13)
    assert r["gini_mean_difference"] == pytest.approx(gmd, rel=1e-13)
    assert r["bimodality_coefficient"] == pytest.approx(bc, rel=1e-12)


def test_er_emd_party_and_trend_recomputed():
    y, p = [0.0, 1.0, 3.0], [0.2, 0.5, 0.3]
    er = sum(p[i] ** 2.0 * p[j] * abs(y[i] - y[j]) for i in range(3) for j in range(3))
    assert esteban_ray_index(y, p, alpha=1.0) == pytest.approx(er, rel=1e-13)
    assert earth_movers_distance([0.0, 1.0, 2.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0, rel=1e-14)
    s = [0.5, 0.3, 0.2]
    h = sum(v * v for v in s)
    r = party_system_indices([50, 30, 20])
    assert (r["fractionalization"], r["effective_number"]) == pytest.approx((1 - h, 1 / h), rel=1e-13)
    t, v = [1.0, 2.0, 3.0, 4.0, 5.0], [0.5, 0.6, 0.8, 0.9, 1.2]
    b = sum((a - 3) * (c - 0.8) for a, c in zip(t, v)) / 10
    assert polarization_trend(t, v)["slope"] == pytest.approx(b, rel=1e-13)
