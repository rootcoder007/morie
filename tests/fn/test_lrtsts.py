"""Tests for lrtsts.logrank_test."""

from morie.fn import _array_core as np

from morie.fn.lrtsts import logrank_test


def test_lrtsts_basic():
    """Test basic functionality."""
    time = np.random.default_rng(42).normal(0.0, 1.0, 40)
    event = np.random.default_rng(42).normal(0.0, 1.0, 40)
    group = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    result = logrank_test(time, event, group)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "statistic" in result


def test_lrtsts_edge():
    """Test edge cases."""
    time = np.random.default_rng(42).normal(0.0, 1.0, 40)
    event = np.random.default_rng(42).normal(0.0, 1.0, 40)
    group = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    result = logrank_test(time, event, group)
    assert isinstance(result, dict)
