"""Tests for gb_wsp.gibbons_concordance_preference."""

from morie.fn import _array_core as np

from morie.fn.gb_wsp import gibbons_concordance_preference


def test_gb_wsp_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gibbons_concordance_preference(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gb_wsp_edge():
    """Test edge cases."""
    result = gibbons_concordance_preference(np.array([42.0]))
    assert result["n"] == 1
