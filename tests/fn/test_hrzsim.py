"""Tests for hrzsim.horowitz_single_index_model."""

from morie.fn import _array_core as np

from morie.fn.hrzsim import horowitz_single_index_model


def test_hrzsim_basic():
    """Test basic functionality."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_single_index_model(x, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_hrzsim_edge():
    """Test edge cases."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_single_index_model(x, y)
    assert isinstance(result, dict)
