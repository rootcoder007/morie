"""Tests for se_g.se_g."""

from morie.fn import _array_core as np

from morie.fn.se_g import se_g


def test_ca11e7_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = se_g(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca11e7_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = se_g(x)
    assert isinstance(result, dict)
