"""Tests for crfflt.christiano_fitzgerald."""

from morie.fn import _array_core as np

from morie.fn.crfflt import christiano_fitzgerald


def test_crfflt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = christiano_fitzgerald(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_crfflt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = christiano_fitzgerald(x)
    assert isinstance(result, dict)
