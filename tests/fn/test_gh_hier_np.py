"""Tests for gh_hier_np.ghosal_hierarchical_np."""

from morie.fn import _array_core as np

from morie.fn.gh_hier_np import ghosal_hierarchical_np


def test_gh_hier_np_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_hierarchical_np(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))


def test_gh_hier_np_edge():
    """Test edge cases."""
    result = ghosal_hierarchical_np(np.array([42.0]))
    assert "K_n" in result
    assert isinstance(result["K_n"], int)
    assert result["K_n"] >= 0
    assert "posterior_positive" in result
    assert "method" in result
