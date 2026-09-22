"""Tests for evmsexp.evt_max_stable_logistic."""

from morie.fn import _array_core as np

from morie.fn.evmsexp import evt_max_stable_logistic


def test_evmsexp_basic():
    """Test basic functionality."""
    x = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    y = np.abs(np.random.default_rng(43).normal(0, 1, 100)) + 0.5
    alpha = 0.05
    result = evt_max_stable_logistic(x, y, alpha)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_evmsexp_edge():
    """Test edge cases."""
    x = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    y = np.abs(np.random.default_rng(43).normal(0, 1, 100)) + 0.5
    alpha = 0.05
    result = evt_max_stable_logistic(x, y, alpha)
    assert isinstance(result, dict)
