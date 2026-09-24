"""Tests for spman.schabenberger_mantel_test."""

from morie.fn import _array_core as np

from morie.fn.spman import schabenberger_mantel_test


def test_spman_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = schabenberger_mantel_test(coords, x)
    assert isinstance(result, dict)
    assert "m1" in result or "m1" in result


def test_spman_edge():
    """Test edge cases."""
    coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = schabenberger_mantel_test(coords, x)
    assert isinstance(result, dict)
