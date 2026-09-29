"""Tests for ksr10.kosorok_m_estimator."""

from morie.fn import _array_core as np
from morie.fn.ksr10 import kosorok_m_estimator


def test_ksr10_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kosorok_m_estimator(x, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ksr10_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kosorok_m_estimator(x, y)
    assert isinstance(result, dict)


def test_huber_location_solves_the_estimating_equation():
    """sum psi_H((x - theta)/eta) = 0 at the returned theta, eta = MAD/0.6745;
    sandwich se = sqrt(mean psi^2 / (mean psi'/eta)^2 / n)."""
    import math

    import pytest

    x = [2.1, 3.4, 1.9, 5.6, 2.8, 3.1, 9.9, 2.5, 3.3, 2.7]
    n = 10
    s = sorted(x)
    med = (s[4] + s[5]) / 2
    dev = sorted(abs(v - med) for v in x)
    eta = (dev[4] + dev[5]) / 2 / 0.6745
    r = kosorok_m_estimator(x)
    th = r["estimate"]
    psi = [max(-1.345, min(1.345, (v - th) / eta)) for v in x]
    assert sum(psi) == pytest.approx(0.0, abs=1e-8)
    A = sum(abs((v - th) / eta) <= 1.345 for v in x) / n / eta
    assert r["se"] == pytest.approx(math.sqrt(sum(p * p for p in psi) / n / A**2 / n), rel=1e-9)
