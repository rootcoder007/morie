"""Tests for spspec.schabenberger_spectral_representation."""

from morie.fn import _array_core as np

from morie.fn.spspec import schabenberger_spectral_representation


def test_spspec_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = schabenberger_spectral_representation(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_spspec_edge():
    """Test edge cases."""
    result = schabenberger_spectral_representation(np.array([42.0]))
    assert result["n"] == 1
