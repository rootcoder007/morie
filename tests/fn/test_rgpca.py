"""Tests for rgpca.rangayyan_pca_signals."""

from morie.fn import _array_core as np

from morie.fn.rgpca import rangayyan_pca_signals


def test_rgpca_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_pca_signals(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_rgpca_edge():
    """Test edge cases."""
    result = rangayyan_pca_signals(np.array([42.0]))
    assert result["n"] == 1
