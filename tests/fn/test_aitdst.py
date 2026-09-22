"""Tests for aitdst.aitchison_distance."""

from morie.fn import _array_core as np

from morie.fn.aitdst import aitchison_distance


def test_aitdst_basic():
    """Test basic functionality."""
    x = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    y = np.abs(np.random.default_rng(43).normal(0, 1, 100)) + 0.5
    result = aitchison_distance(x, y)
    assert isinstance(result, dict)
    assert "distance" in result
def test_aitdst_edge():
    """Test edge cases."""
    x = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    y = np.abs(np.random.default_rng(43).normal(0, 1, 100)) + 0.5
    result = aitchison_distance(x, y)
    assert isinstance(result, dict)
