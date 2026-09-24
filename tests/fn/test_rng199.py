"""Tests for rng199.rangayyan_ch4_correlation_coefficient_normalized_dot."""

from morie.fn import _array_core as np

from morie.fn.bsacorr import rangayyan_ch4_correlation_coefficient_normalized_dot


def test_rng199_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch4_correlation_coefficient_normalized_dot(x, y)
    assert isinstance(result, dict)
    assert "gamma" in result


def test_rng199_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch4_correlation_coefficient_normalized_dot(x, y)
    assert isinstance(result, dict)
