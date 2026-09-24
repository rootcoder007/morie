"""Tests for hrzasym.horowitz_one_step_efficient."""

from morie.fn import _array_core as np

from morie.fn.hrzasym import horowitz_one_step_efficient


def test_hrzasym_basic():
    """Test basic functionality."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_one_step_efficient(x, y)
    assert isinstance(result, dict)
    assert "beta" in result


def test_hrzasym_edge():
    """Test edge cases."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_one_step_efficient(x, y)
    assert isinstance(result, dict)
