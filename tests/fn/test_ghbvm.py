"""Tests for ghbvm.ghosal_bernstein_von_mises."""

from morie.fn import _array_core as np

from morie.fn.ghbvm import ghosal_bernstein_von_mises


def test_ghbvm_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_bernstein_von_mises(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_ghbvm_edge():
    """Test edge cases."""
    result = ghosal_bernstein_von_mises(np.array([42.0]))
    assert result["n"] == 1
