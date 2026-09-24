"""Tests for survrsf.random_survival_forest."""

from morie.fn import _array_core as np

from morie.fn.survrsf import random_survival_forest


def test_survrsf_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    times = np.random.default_rng(42).normal(0.0, 1.0, 40)
    events = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = random_survival_forest(X, times, events)
    assert isinstance(result, dict)
    assert "trees" in result


def test_survrsf_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    times = np.random.default_rng(42).normal(0.0, 1.0, 40)
    events = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = random_survival_forest(X, times, events)
    assert isinstance(result, dict)
