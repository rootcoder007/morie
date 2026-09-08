"""Tests for fisher_z.fisher_z."""

from morie.fn import _array_core as np

from morie.fn.fisher_z import fisher_z


def test_ca11e12_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = fisher_z(x)
    assert isinstance(result, dict)
    assert "statistic" in result or "estimate" in result


def test_ca11e12_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = fisher_z(x)
    assert isinstance(result, dict)
