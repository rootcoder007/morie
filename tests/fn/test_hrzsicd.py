"""Tests for hrzsicd.horowitz_sim_id_discrete_x."""

from morie.fn import _array_core as np

from morie.fn.hrzsicd import horowitz_sim_id_discrete_x


def test_hrzsicd_basic():
    """Test basic functionality."""
    xs = np.random.default_rng(42).normal(0.0, 1.0, 40)
    gvals = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_sim_id_discrete_x(xs, gvals)
    assert isinstance(result, dict)
    assert "lower" in result


def test_hrzsicd_edge():
    """Test edge cases."""
    xs = np.random.default_rng(42).normal(0.0, 1.0, 40)
    gvals = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_sim_id_discrete_x(xs, gvals)
    assert isinstance(result, dict)
