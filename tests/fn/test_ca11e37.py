"""Tests for ca11e37.ca_chapter_11_equation_37."""

from morie.fn import _array_core as np

from morie.fn.ca11e37 import ca_chapter_11_equation_37


def test_ca11e37_basic():
    """Test basic functionality."""
    ys = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    ws = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = ca_chapter_11_equation_37(ys, ws)
    assert isinstance(result, dict)
    assert "mean" in result
def test_ca11e37_edge():
    """Test edge cases."""
    ys = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    ws = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = ca_chapter_11_equation_37(ys, ws)
    assert isinstance(result, dict)
