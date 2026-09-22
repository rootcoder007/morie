"""Tests for gb731.gibbons_linrank_moments."""

from morie.fn import _array_core as np

from morie.fn.gb731 import gibbons_linrank_moments


def test_gb731_basic():
    """Test basic functionality."""
    a = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    result = gibbons_linrank_moments(a, m)
    assert isinstance(result, dict)
    assert "mean" in result
def test_gb731_edge():
    """Test edge cases."""
    a = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    result = gibbons_linrank_moments(a, m)
    assert isinstance(result, dict)
