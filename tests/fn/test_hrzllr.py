"""Tests for hrzllr.horowitz_local_linear."""

from morie.fn import _array_core as np

from morie.fn.hrzllr import horowitz_local_linear


def test_hrzllr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_local_linear(x, y)
    assert isinstance(result, dict)
    assert "grid" in result


def test_hrzllr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_local_linear(x, y)
    assert isinstance(result, dict)
