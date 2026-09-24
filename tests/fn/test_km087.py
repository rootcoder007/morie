"""Tests for km087.kamath_ch6_cps_metric."""

from morie.fn import _array_core as np

from morie.fn.km087 import kamath_ch6_cps_metric


def test_km087_basic():
    """Test basic functionality."""
    U = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    M = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kamath_ch6_cps_metric(U, M)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_km087_edge():
    """Test edge cases."""
    U = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    M = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kamath_ch6_cps_metric(U, M)
    assert isinstance(result, dict)
