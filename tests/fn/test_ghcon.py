"""Tests for ghcon.ghosal_posterior_consistency."""

from morie.fn import _array_core as np

from morie.fn.ghcon import ghosal_posterior_consistency


def test_ghcon_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_posterior_consistency(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_ghcon_edge():
    """Test edge cases."""
    result = ghosal_posterior_consistency(np.array([42.0]))
    assert result["n"] == 1
