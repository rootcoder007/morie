"""Tests for se_d_from_se_r.se_d_from_se_r."""

from morie.fn import _array_core as np

from morie.fn.se_d_from_se_r import se_d_from_se_r


def test_ca11e23_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = se_d_from_se_r(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca11e23_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = se_d_from_se_r(x)
    assert isinstance(result, dict)
