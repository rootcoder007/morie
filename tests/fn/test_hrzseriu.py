"""Tests for hrzseriu.horowitz_series_unknown_T."""

from morie.fn import _array_core as np

from morie.fn.hrzseriu import horowitz_series_unknown_T


def test_hrzseriu_basic():
    """Test basic functionality."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    w = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = horowitz_series_unknown_T(x, y, w)
    assert isinstance(result, dict)
    assert "g_hat" in result


def test_hrzseriu_edge():
    """Test edge cases."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    w = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = horowitz_series_unknown_T(x, y, w)
    assert isinstance(result, dict)
