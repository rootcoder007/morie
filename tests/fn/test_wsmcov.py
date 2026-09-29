"""Tests for wsmcov.wasserman_covariance."""

from morie.fn import _array_core as np
from morie.fn.wsmcov import wasserman_covariance


def test_wsmcov_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = wasserman_covariance(x, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_wsmcov_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = wasserman_covariance(x, y)
    assert isinstance(result, dict)


def test_covariance_divisors_recomputed():
    import pytest

    x = [1.0, 2.5, 3.0, 4.5]
    y = [2.0, 2.0, 5.0, 4.0]
    mx, my = sum(x) / 4, sum(y) / 4
    c = sum((a - mx) * (b - my) for a, b in zip(x, y)) / 4
    r = wasserman_covariance(x, y)
    assert r["estimate"] == pytest.approx(c, rel=1e-13)
    assert r["sample_covariance"] == pytest.approx(c * 4 / 3, rel=1e-13)
    assert r["estimate"] == pytest.approx(sum(a * b for a, b in zip(x, y)) / 4 - mx * my, rel=1e-12)
