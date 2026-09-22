"""Tests for bivand20138e4.bivand2013_chapter_8_equation_4."""

from morie.fn import _array_core as np

from morie.fn.bivand20138e4 import bivand2013_chapter_8_equation_4


def test_bivand20138e4_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).normal(0, 1, 100)
    z = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    result = bivand2013_chapter_8_equation_4(coords, z)
    assert isinstance(result, dict)
    assert "gamma" in result
def test_bivand20138e4_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).normal(0, 1, 100)
    z = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    result = bivand2013_chapter_8_equation_4(coords, z)
    assert isinstance(result, dict)
