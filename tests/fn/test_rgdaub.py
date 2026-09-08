"""Tests for rgdaub.rangayyan_daubechies."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_daubechies


def test_rgdaub_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_daubechies(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_rgdaub_edge():
    """Test edge cases."""
    result = rangayyan_daubechies(np.array([42.0]))
    assert result["n"] == 1
