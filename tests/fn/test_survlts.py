"""Tests for survlts.life_table_smoothed."""

from morie.fn import _array_core as np

from morie.fn.survlts import life_table_smoothed


def test_survlts_basic():
    """Test basic functionality."""
    time = np.random.default_rng(42).normal(0.0, 1.0, 40)
    event = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    result = life_table_smoothed(time, event)
    assert isinstance(result, dict)
    assert "grid" in result


def test_survlts_edge():
    """Test edge cases."""
    time = np.random.default_rng(42).normal(0.0, 1.0, 40)
    event = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    result = life_table_smoothed(time, event)
    assert isinstance(result, dict)
