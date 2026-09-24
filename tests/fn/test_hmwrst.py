"""Tests for hmwrst.geron_warm_restarts."""

from morie.fn import _array_core as np

from morie.fn.hmwrst import geron_warm_restarts


def test_hmwrst_basic():
    """Test basic functionality."""
    t = np.array([float(i + 1) for i in range(40)])
    result = geron_warm_restarts(t)
    assert isinstance(result, dict)
    assert "estimate" in result or "eta" in result


def test_hmwrst_edge():
    """Test edge cases."""
    t = np.array([float(i + 1) for i in range(40)])
    result = geron_warm_restarts(t)
    assert isinstance(result, dict)
