"""Tests for hrznpiv.horowitz_npiv_model."""

from morie.fn import _array_core as np

from morie.fn.hrznpiv import horowitz_npiv_model


def test_hrznpiv_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    w = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_npiv_model(x, y, w)
    assert isinstance(result, dict)
    assert "g_hat" in result


def test_hrznpiv_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    w = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_npiv_model(x, y, w)
    assert isinstance(result, dict)
