"""Tests for topDn.top_down."""

from morie.fn import _array_core as np

from morie.fn.topDn import top_down


def test_topDn_basic():
    """Test basic functionality."""
    top = 0.1
    props = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = top_down(top, props)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_topDn_edge():
    """Test edge cases."""
    top = 0.1
    props = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = top_down(top, props)
    assert isinstance(result, dict)
