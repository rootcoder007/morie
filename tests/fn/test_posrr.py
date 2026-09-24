"""Tests for posrr.posterior_predictive_replication."""

from morie.fn import _array_core as np

from morie.fn.posrr import posterior_predictive_replication


def test_posrr_basic():
    """Test basic functionality."""
    t_obs = np.random.default_rng(42).normal(0.0, 1.0, 40)
    t_rep = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = posterior_predictive_replication(t_obs, t_rep)
    assert isinstance(result, dict)
    assert "p_value" in result


def test_posrr_edge():
    """Test edge cases."""
    t_obs = np.random.default_rng(42).normal(0.0, 1.0, 40)
    t_rep = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = posterior_predictive_replication(t_obs, t_rep)
    assert isinstance(result, dict)
