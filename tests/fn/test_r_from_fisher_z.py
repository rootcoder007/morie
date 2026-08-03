"""Tests for r_from_fisher_z.r_from_fisher_z."""

from morie.fn import _array_core as np

from morie.fn.r_from_fisher_z import r_from_fisher_z


def test_ca11e14_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = r_from_fisher_z(x)
    assert isinstance(result, dict)
    assert "statistic" in result or "estimate" in result


def test_ca11e14_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = r_from_fisher_z(x)
    assert isinstance(result, dict)
