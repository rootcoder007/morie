"""Tests for ksr07.kosorok_bootstrap_empirical."""

from morie.fn import _array_core as np

from morie.fn.ksr07 import kosorok_bootstrap_empirical


def test_ksr07_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = kosorok_bootstrap_empirical(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))


def test_ksr07_edge():
    """Test edge cases."""
    x = np.array([1.0, 2.0])
    result = kosorok_bootstrap_empirical(x)
    assert result["n"] == 2
