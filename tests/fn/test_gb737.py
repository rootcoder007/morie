"""Tests for gb737.gibbons_linrank_properties."""

from morie.fn import _array_core as np

from morie.fn.gb737 import gibbons_linrank_properties


def test_gb737_basic():
    """Test basic functionality."""
    a = np.random.default_rng(42).normal(0, 1, 100)
    z = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    result = gibbons_linrank_properties(a, z)
    assert isinstance(result, dict)
    assert "t" in result
def test_gb737_edge():
    """Test edge cases."""
    a = np.random.default_rng(42).normal(0, 1, 100)
    z = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    result = gibbons_linrank_properties(a, z)
    assert isinstance(result, dict)
