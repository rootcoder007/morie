"""Tests for ghcrt.ghosal_contraction_rate."""

from morie.fn import _array_core as np

from morie.fn.ghcrt import ghosal_contraction_rate


def test_ghcrt_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_contraction_rate(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_ghcrt_edge():
    """Test edge cases."""
    result = ghosal_contraction_rate(np.array([42.0]))
    assert result["n"] == 1
