"""Tests for d_from_r_pointbiserial.d_from_r_pointbiserial."""

from morie.fn import _array_core as np

from morie.fn.d_from_r_pointbiserial import d_from_r_pointbiserial


def test_ca11e22_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = d_from_r_pointbiserial(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca11e22_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = d_from_r_pointbiserial(x)
    assert isinstance(result, dict)
