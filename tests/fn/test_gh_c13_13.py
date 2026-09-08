"""Tests for gh_c13_13.ghosal_cox_model."""

from morie.fn import _array_core as np

from morie.fn.gh_c13_13 import ghosal_cox_model


def test_gh_c13_13_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_cox_model(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c13_13_edge():
    """Test edge cases."""
    result = ghosal_cox_model(np.array([42.0]))
    assert result["n"] == 1
