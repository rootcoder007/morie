"""Tests for se_log_rr.se_log_rr."""

from morie.fn import _array_core as np

from morie.fn.se_log_rr import se_log_rr


def test_ca11e9_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = se_log_rr(x)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_ca11e9_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = se_log_rr(x)
    assert isinstance(result, dict)
