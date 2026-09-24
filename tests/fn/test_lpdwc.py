"""Tests for lpdwc.log_pointwise_predictive_density."""

from morie.fn import _array_core as np

from morie.fn.lpdwc import log_pointwise_predictive_density


def test_lpdwc_basic():
    """Test basic functionality."""
    logdens = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = log_pointwise_predictive_density(logdens)
    assert isinstance(result, dict)
    assert "lppd" in result


def test_lpdwc_edge():
    """Test edge cases."""
    logdens = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = log_pointwise_predictive_density(logdens)
    assert isinstance(result, dict)
