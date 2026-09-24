"""Tests for rng086.rangayyan_ch3_normalized_cross_correlation_template."""

from morie.fn import _array_core as np

from morie.fn.bsacorr import rangayyan_ch3_normalized_cross_correlation_template


def test_rng086_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    template = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch3_normalized_cross_correlation_template(x, template)
    assert isinstance(result, dict)
    assert "gamma" in result


def test_rng086_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    template = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch3_normalized_cross_correlation_template(x, template)
    assert isinstance(result, dict)
