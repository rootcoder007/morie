"""Tests for bivand20138e5.bivand2013_chapter_8_equation_5."""

from morie.fn import _array_core as np

from morie.fn.bivand20138e5 import bivand2013_chapter_8_equation_5


def test_bivand20138e5_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 3))
    z = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    result = bivand2013_chapter_8_equation_5(X, z)
    assert isinstance(result, dict)
    assert "beta" in result
def test_bivand20138e5_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 3))
    z = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    result = bivand2013_chapter_8_equation_5(X, z)
    assert isinstance(result, dict)
