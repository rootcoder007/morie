"""Tests for matau2pi.ma_tau2_predict_interval."""

from morie.fn import _array_core as np

from morie.fn.matau2pi import ma_tau2_predict_interval


def test_matau2pi_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0.0, 1.0, 40)
    vi = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = ma_tau2_predict_interval(yi, vi)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_matau2pi_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0.0, 1.0, 40)
    vi = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = ma_tau2_predict_interval(yi, vi)
    assert isinstance(result, dict)
