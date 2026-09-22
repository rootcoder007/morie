"""Tests for gh_ppt_consist.ghosal_polya_tree_consist_rate."""

from morie.fn import _array_core as np

from morie.fn.gh_ppt_consist import ghosal_polya_tree_consist_rate


def test_gh_ppt_consist_basic():
    """Test basic functionality."""
    result = ghosal_polya_tree_consist_rate(ns=(50, 200, 800), depth=6, seed=42)
    # Function documents returning a payload with keys:
    # "estimate", "l1_error_by_n", "contracting", "method"
    assert "estimate" in result
    assert "l1_error_by_n" in result
    assert "contracting" in result
    assert "method" in result
    # estimate should be a finite float (L1 error on a query grid)
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    # l1_error_by_n should be a sequence of finite floats, one per n in ns
    l1_by_n = list(result["l1_error_by_n"])
    assert len(l1_by_n) == 3
    assert np.all(np.isfinite(np.asarray(l1_by_n, dtype=float)))
    # contracting should be a boolean
    assert isinstance(result["contracting"], (bool, np.bool_))


def test_gh_ppt_consist_edge():
    """Test edge cases."""
    # Single sample size ns as a 1-tuple
    result = ghosal_polya_tree_consist_rate(ns=(42,), depth=6, seed=42)
    # Function returns "l1_error_by_n" (a sequence), not "n"
    assert "l1_error_by_n" in result
    l1_by_n = list(result["l1_error_by_n"])
    assert len(l1_by_n) == 1
    assert np.all(np.isfinite(np.asarray(l1_by_n, dtype=float)))
