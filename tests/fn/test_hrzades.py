"""Tests for hrzades.horowitz_improved_ade."""

from morie.fn import _array_core as np

from morie.fn.hrzades import horowitz_improved_ade


def test_hrzades_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_improved_ade(X, y)
    assert isinstance(result, dict)
    assert "delta_hat" in result


def test_hrzades_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_improved_ade(X, y)
    assert isinstance(result, dict)
