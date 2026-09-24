"""Tests for km099.kamath_ch6_emt_metric."""

from morie.fn import _array_core as np

from morie.fn.km099 import kamath_ch6_emt_metric


def test_km099_basic():
    """Test basic functionality."""
    Yhat = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    c = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = kamath_ch6_emt_metric(Yhat, c)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_km099_edge():
    """Test edge cases."""
    Yhat = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    c = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = kamath_ch6_emt_metric(Yhat, c)
    assert isinstance(result, dict)
