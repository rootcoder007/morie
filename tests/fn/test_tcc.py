"""Tests for tcc.test_characteristic_curve (IRT true score)."""

import math

import pytest

from morie.fn.tcc import test_characteristic_curve as tcc_fn


def _p(t, a, b, c, u, D):
    return c + (u - c) / (1 + math.exp(-D * a * (t - b)))


def test_tcc_basic():
    """T(theta) = sum_i P_i(theta) under the 4PL; floor sum c, ceiling
    sum u; T is increasing in theta."""
    a, b, c, u = [1.2, 0.8, 1.5], [-0.5, 0.3, 1.1], [0.2, 0.0, 0.1], [1.0, 0.95, 1.0]
    th = [-2.0, 0.0, 1.5]
    r = tcc_fn(th, a, b, c, u, D=1.702)
    for t, v in zip(th, r["tcc"]):
        assert v == pytest.approx(sum(_p(t, *z, 1.702) for z in zip(a, b, c, u)), abs=1e-15)
    assert r["floor"] == pytest.approx(0.3, abs=1e-15)
    assert r["ceiling"] == pytest.approx(2.95, abs=1e-15)
    assert r["tcc"][0] < r["tcc"][1] < r["tcc"][2]


def test_tcc_edge():
    """At theta = b a 2PL item contributes exactly 1/2; a scalar theta
    returns a scalar; c >= upper raises."""
    r = tcc_fn(0.3, [1.0, 2.0], [0.3, 0.3])
    assert r["tcc"] == pytest.approx(1.0, abs=1e-15)
    with pytest.raises(ValueError):
        tcc_fn(0.0, [1.0], [0.0], c=[0.5], upper=[0.5])
