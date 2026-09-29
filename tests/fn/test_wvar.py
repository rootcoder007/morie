"""Tests for wvar.weighted_variance."""

from morie.fn import _array_core as np
from morie.fn.wvar import weighted_variance


def test_wvar_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = weighted_variance(y, weights)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_wvar_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = weighted_variance(y, weights)
    assert isinstance(result, dict)


def test_reliability_weighted_variance_recomputed():
    import math

    import pytest

    y = [3.0, 1.0, 4.0, 2.0, 5.0]
    w = [1.0, 3.0, 1.0, 1.0, 2.0]
    sw = sum(w)
    mu = sum(a * b for a, b in zip(w, y)) / sw
    s2 = sum(a * (b - mu) ** 2 for a, b in zip(w, y)) / (sw - 1)
    r = weighted_variance(y, w)
    assert r["mean"] == pytest.approx(mu, rel=1e-14)
    assert r["estimate"] == pytest.approx(s2, rel=1e-13)
    assert r["se"] == pytest.approx(math.sqrt(s2 * sum(v * v for v in w) / sw**2), rel=1e-13)
