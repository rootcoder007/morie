"""Tests for fzecdf.fauzi_ecdf."""

from morie.fn import _array_core as np

from morie.fn.fzecdf import fauzi_ecdf


def test_fzecdf_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fauzi_ecdf(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_fzecdf_edge():
    """Test edge cases."""
    result = fauzi_ecdf(np.array([42.0]))
    assert result["n"] == 1
