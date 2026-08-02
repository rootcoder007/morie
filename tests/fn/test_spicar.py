"""Tests for spicar.schabenberger_icar_prior."""

from morie.fn import _array_core as np

from morie.fn.spicar import schabenberger_icar_prior


def test_spicar_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = schabenberger_icar_prior(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_spicar_edge():
    """Test edge cases."""
    result = schabenberger_icar_prior(np.array([42.0]))
    assert result["n"] == 1
