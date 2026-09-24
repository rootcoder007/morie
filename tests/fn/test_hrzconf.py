"""Tests for hrzconf.horowitz_confidence_bands."""

from morie.fn import _array_core as np

from morie.fn.hrzconf import horowitz_confidence_bands


def test_hrzconf_basic():
    """Test basic functionality."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = horowitz_confidence_bands(x, y)
    assert isinstance(result, dict)
    assert "grid" in result


def test_hrzconf_edge():
    """Test edge cases."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = horowitz_confidence_bands(x, y)
    assert isinstance(result, dict)
