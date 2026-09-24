"""Tests for km089.kamath_ch6_sgs_invariance."""

from morie.fn import _array_core as np

from morie.fn.km089 import kamath_ch6_sgs_invariance


def test_km089_basic():
    """Test basic functionality."""
    Yhat_i = np.random.default_rng(42).normal(0.0, 1.0, 40)
    Yhat_j = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = kamath_ch6_sgs_invariance(Yhat_i, Yhat_j)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_km089_edge():
    """Test edge cases."""
    Yhat_i = np.random.default_rng(42).normal(0.0, 1.0, 40)
    Yhat_j = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = kamath_ch6_sgs_invariance(Yhat_i, Yhat_j)
    assert isinstance(result, dict)
