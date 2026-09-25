"""Tests for inlasm.inla_spatial."""

import math

import pytest

from morie.fn.inlasm import gaussian_approximation


def test_inlasm_basic():
    """A Gaussian likelihood makes the approximation exact: mode (tau0 mu0 +
    y / s^2) / (tau0 + 1 / s^2) and precision tau0 + 1 / s^2."""
    y, s2, mu0, tau0 = 2.3, 0.5, -1.0, 0.8
    r = gaussian_approximation(lambda x: -(y - x) ** 2 / (2 * s2), lambda x: (y - x) / s2,
                               lambda x: -1 / s2, mu0, tau0)
    assert r["mode"] == pytest.approx((tau0 * mu0 + y / s2) / (tau0 + 1 / s2), rel=1e-12)
    assert r["precision"] == pytest.approx(tau0 + 1 / s2, rel=1e-12)


def test_inlasm_edge():
    """A Poisson log-rate likelihood: the mode zeros the gradient of the log
    posterior, and the precision is its negative curvature there."""
    y = 4.0
    ll = lambda x: y * x - math.exp(x)
    d1 = lambda x: y - math.exp(x)
    d2 = lambda x: -math.exp(x)
    r = gaussian_approximation(ll, d1, d2, 0.0, 1.0)
    m = r["mode"]
    assert d1(m) - 1.0 * (m - 0.0) == pytest.approx(0.0, abs=1e-10)
    assert r["precision"] == pytest.approx(math.exp(m) + 1.0, rel=1e-10)


