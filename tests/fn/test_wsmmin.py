"""Tests for wsmmin.wasserman_minimax."""

from morie.fn import _array_core as np

from morie.fn.wsmmin import wasserman_minimax


def test_wsmmin_basic():
    """Test basic functionality."""
    loss = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    estimator = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    family = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = wasserman_minimax(loss, estimator, family)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_wsmmin_edge():
    """Test edge cases."""
    loss = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    estimator = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    family = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = wasserman_minimax(loss, estimator, family)
    assert isinstance(result, dict)
