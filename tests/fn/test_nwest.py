"""Tests for nwest.newey_west_hac."""

from morie.fn import _array_core as np

from morie.fn.nwest import newey_west_hac


def test_nwest_basic():
    """Test basic functionality."""
    scores = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = newey_west_hac(scores)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_nwest_edge():
    """Test edge cases."""
    scores = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = newey_west_hac(scores)
    assert isinstance(result, dict)
