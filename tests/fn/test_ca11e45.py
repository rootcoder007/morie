"""Tests for ca11e45.ca_chapter_11_equation_45."""

from morie.fn import _array_core as np

from morie.fn.ca11e45 import ca_chapter_11_equation_45


def test_ca11e45_basic():
    """Test basic functionality."""
    ys_by_group = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    ws_by_group = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = ca_chapter_11_equation_45(ys_by_group, ws_by_group)
    assert isinstance(result, dict)
    assert "q_within" in result
def test_ca11e45_edge():
    """Test edge cases."""
    ys_by_group = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    ws_by_group = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = ca_chapter_11_equation_45(ys_by_group, ws_by_group)
    assert isinstance(result, dict)
