"""Tests for se_fisher_z.se_fisher_z."""

from morie.fn import _array_core as np

from morie.fn.se_fisher_z import se_fisher_z


def test_ca11e13_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = se_fisher_z(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca11e13_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = se_fisher_z(x)
    assert isinstance(result, dict)
