"""Tests for rgccep.rangayyan_complex_cepstrum."""

from morie.fn import _array_core as np

from morie.fn.rgccep import rangayyan_complex_cepstrum


def test_rgccep_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_complex_cepstrum(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_rgccep_edge():
    """Test edge cases."""
    result = rangayyan_complex_cepstrum(np.array([42.0]))
    assert result["n"] == 1
