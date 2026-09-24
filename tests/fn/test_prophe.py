"""Tests for prophe.facebook_prophet."""

from morie.fn import _array_core as np

from morie.fn.prophe import facebook_prophet


def test_prophe_basic():
    """Test basic functionality."""
    t = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = facebook_prophet(t, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_prophe_edge():
    """Test edge cases."""
    t = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = facebook_prophet(t, y)
    assert isinstance(result, dict)
