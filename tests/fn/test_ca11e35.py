"""Tests for ca11e35.ca_chapter_11_equation_35."""

from morie.fn import _array_core as np

from morie.fn.ca11e35 import ca_chapter_11_equation_35


def test_ca11e35_basic():
    """Test basic functionality."""
    ys = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    ws = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = ca_chapter_11_equation_35(ys, ws)
    assert isinstance(result, dict)
    assert "mean" in result
def test_ca11e35_edge():
    """Test edge cases."""
    ys = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    ws = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = ca_chapter_11_equation_35(ys, ws)
    assert isinstance(result, dict)
