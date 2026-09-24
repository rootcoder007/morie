"""Tests for hrzphd.horowitz_ph_discrete_obs."""

from morie.fn import _array_core as np
from morie.fn.hrzphd import horowitz_ph_discrete_obs


def test_hrzphd_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    K = 5
    t_discrete = rng.integers(1, K + 1, n)
    x = rng.normal(0, 1, (n, p))
    event = rng.integers(0, 2, n)
    result = horowitz_ph_discrete_obs(t_discrete, x, event, K=K)
    assert isinstance(result, dict)
    assert "beta_hat" in result
    assert "h_j_hat" in result
    assert "theta_hat" in result


def test_hrzphd_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    K = 5
    t_discrete = rng.integers(1, K + 1, n)
    x = rng.normal(0, 1, (n, p))
    # event omitted -> defaults to all interval events
    result = horowitz_ph_discrete_obs(t_discrete, x, K=K)
    assert isinstance(result, dict)
    assert "beta_hat" in result
