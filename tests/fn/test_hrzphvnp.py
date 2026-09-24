"""Tests for hrzphvnp.horowitz_ph_frailty_nonpar."""

from morie.fn import _array_core as np

from morie.fn.hrzphvnp import horowitz_ph_frailty_nonpar


def test_hrzphvnp_basic():
    """Test basic functionality."""
    t = np.array([float(i + 1) for i in range(40)])
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = horowitz_ph_frailty_nonpar(t, x)
    assert isinstance(result, dict)
    assert "beta_hat" in result


def test_hrzphvnp_edge():
    """Test edge cases."""
    t = np.array([float(i + 1) for i in range(40)])
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = horowitz_ph_frailty_nonpar(t, x)
    assert isinstance(result, dict)
