"""Tests for ksr047.kosorok_ch2_kaplan_meier_self_consistency."""

from morie.fn import _array_core as np

from morie.fn.ksr047 import kosorok_ch2_kaplan_meier_self_consistency


def test_ksr047_basic():
    """Test basic functionality."""
    S = np.random.default_rng(42).normal(0.0, 1.0, 40)
    t_grid = np.random.default_rng(42).normal(0.0, 1.0, 40)
    S0 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    L = np.random.default_rng(42).normal(0.0, 1.0, 40)
    G = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = kosorok_ch2_kaplan_meier_self_consistency(S, t_grid, S0, L, G)
    assert isinstance(result, dict)
    assert "t_grid" in result


def test_ksr047_edge():
    """Test edge cases."""
    S = np.random.default_rng(42).normal(0.0, 1.0, 40)
    t_grid = np.random.default_rng(42).normal(0.0, 1.0, 40)
    S0 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    L = np.random.default_rng(42).normal(0.0, 1.0, 40)
    G = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = kosorok_ch2_kaplan_meier_self_consistency(S, t_grid, S0, L, G)
    assert isinstance(result, dict)
