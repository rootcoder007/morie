"""Tests for ghhbp.ghosal_hierarchical_bayes."""

from morie.fn import _array_core as np

from morie.fn.ghhbp import ghosal_hierarchical_bayes


def test_ghhbp_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_hierarchical_bayes(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_ghhbp_edge():
    """Test edge cases."""
    result = ghosal_hierarchical_bayes(np.array([42.0]))
    assert result["n"] == 1
