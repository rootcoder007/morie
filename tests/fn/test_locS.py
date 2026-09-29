"""Tests for locS.location_scale_estimator."""

from morie.fn import _array_core as np
from morie.fn.locS import location_scale_estimator


def test_locS_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = location_scale_estimator(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_locS_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = location_scale_estimator(x)
    assert isinstance(result, dict)


def test_huber_proposal_2_sweeps_replayed():
    """MASS::hubers' iteration: Winsorise at mu +- k s, mean, s from the
    Winsorised sum of squares over (n - 1) beta; returns the pre-update iterate."""
    import math

    import pytest

    x = [2.1, 3.4, 1.9, 5.6, 2.8, 3.1, 9.9, 2.5, 3.3, 2.7]
    n, k = 10, 1.5
    th = math.erf(k / math.sqrt(2))
    beta = th + k * k * (1 - th) - 2 * k * math.exp(-k * k / 2) / math.sqrt(2 * math.pi)
    s = sorted(x)
    mu0 = (s[4] + s[5]) / 2
    d = sorted(abs(v - mu0) for v in x)
    s0 = 1.4826 * (d[4] + d[5]) / 2
    for _ in range(30):
        yy = [min(max(mu0 - k * s0, v), mu0 + k * s0) for v in x]
        mu1 = sum(yy) / n
        s1 = math.sqrt(sum((v - mu1) ** 2 for v in yy) / (n - 1) / beta)
        if abs(mu0 - mu1) < 1e-6 * s0 and abs(s0 - s1) < 1e-6 * s0:
            break
        mu0, s0 = mu1, s1
    r = location_scale_estimator(x)
    assert r["estimate"] == pytest.approx(mu0, rel=1e-12)
    assert r["scale"] == pytest.approx(s0, rel=1e-12)
    assert r["beta"] == pytest.approx(beta, rel=1e-12)
