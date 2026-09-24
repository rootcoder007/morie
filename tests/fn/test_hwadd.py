"""Tests for hwadd.holt_winters_additive."""

from morie.fn import _array_core as np

from morie.fn.hwadd import holt_winters_additive


def test_hwadd_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = holt_winters_additive(y)
    assert isinstance(result, dict)
    assert "forecast" in result


def test_hwadd_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = holt_winters_additive(y)
    assert isinstance(result, dict)
