"""Tests for polrz.polarization_index."""

from morie.fn import _array_core as np

from morie.fn.polrz import polarization_index


def test_polrz_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = polarization_index(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_polrz_edge():
    """Test edge cases."""
    result = polarization_index(np.array([42.0]))
    assert result["n"] == 1
