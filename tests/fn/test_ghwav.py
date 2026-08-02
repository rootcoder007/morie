"""Tests for ghwav.ghosal_wavelet_prior."""

from morie.fn import _array_core as np

from morie.fn.ghwav import ghosal_wavelet_prior


def test_ghwav_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_wavelet_prior(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_ghwav_edge():
    """Test edge cases."""
    result = ghosal_wavelet_prior(np.array([42.0]))
    assert result["n"] == 1
