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


def test_ppp_is_paired_by_draw():
    import pytest

    t_obs = [1.0, 2.0, 3.0, 4.0]
    t_rep = [1.5, 1.0, 3.5, 3.0]
    r = posterior_predictive_replication(t_obs, t_rep)
    assert r["p_value"] == pytest.approx(2 / 4, rel=1e-15)
    assert r["p_two_sided"] == pytest.approx(1.0, rel=1e-15)
    assert posterior_predictive_replication(2.0, t_rep)["n_extreme"] == 2.0
