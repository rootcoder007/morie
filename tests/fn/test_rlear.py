"""Tests for rlear.r_learner."""

from morie.fn import _array_core as np

from morie.fn.rlear import r_learner


def test_rlear_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    t = np.array([float(i + 1) for i in range(40)])
    m = np.random.default_rng(42).normal(0.0, 1.0, 40)
    e = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = r_learner(y, t, m, e)
    assert isinstance(result, dict)
    assert "tau" in result


def test_rlear_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    t = np.array([float(i + 1) for i in range(40)])
    m = np.random.default_rng(42).normal(0.0, 1.0, 40)
    e = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = r_learner(y, t, m, e)
    assert isinstance(result, dict)
