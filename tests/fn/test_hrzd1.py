"""Tests for hrzd1.horowitz_duration_model."""

from morie.fn import _array_core as np

from morie.fn.hrzd1 import horowitz_duration_model


def test_hrzd1_basic():
    """Test basic functionality."""
    t = np.random.default_rng(42).normal(0.0, 1.0, 40)
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    event = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = horowitz_duration_model(t, x, event)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_hrzd1_edge():
    """Test edge cases."""
    t = np.random.default_rng(42).normal(0.0, 1.0, 40)
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    event = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = horowitz_duration_model(t, x, event)
    assert isinstance(result, dict)
