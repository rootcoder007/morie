"""Tests for rgminph.rangayyan_min_phase."""

from morie.fn import _array_core as np

from morie.fn.rgminph import rangayyan_min_phase


def test_rgminph_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_min_phase(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_rgminph_edge():
    """Test edge cases."""
    result = rangayyan_min_phase(np.array([42.0]))
    assert result["n"] == 1
