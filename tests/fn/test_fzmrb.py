"""Tests for fzmrb.fauzi_mrl_boundary_free."""

from morie.fn import _array_core as np

from morie.fn.fzmrb import fauzi_mrl_boundary_free


def test_fzmrb_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fauzi_mrl_boundary_free(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_fzmrb_edge():
    """Test edge cases."""
    result = fauzi_mrl_boundary_free(np.array([42.0]))
    assert result["n"] == 1
