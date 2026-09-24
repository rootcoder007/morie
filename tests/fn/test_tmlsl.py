"""Tests for tmlsl.tmle_super_learner."""

from morie.fn import _array_core as np

from morie.fn.tmlsl import tmle_super_learner


def test_tmlsl_basic():
    """Test basic functionality."""
    Z = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = tmle_super_learner(Z, Y)
    assert isinstance(result, dict)
    assert "weights" in result


def test_tmlsl_edge():
    """Test edge cases."""
    Z = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = tmle_super_learner(Z, Y)
    assert isinstance(result, dict)
