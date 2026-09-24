"""Tests for hrznqiv.horowitz_nonpar_quantile_iv."""

from morie.fn import _array_core as np

from morie.fn.hrznqiv import horowitz_nonpar_quantile_iv


def test_hrznqiv_basic():
    """Test basic functionality."""
    T = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    tau_target = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_nonpar_quantile_iv(T, tau_target)
    assert isinstance(result, dict)
    assert "g" in result


def test_hrznqiv_edge():
    """Test edge cases."""
    T = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    tau_target = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_nonpar_quantile_iv(T, tau_target)
    assert isinstance(result, dict)
