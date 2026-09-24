"""Tests for tmlvts.tmle_var_targeting."""

from morie.fn import _array_core as np

from morie.fn.tmlvts import tmle_var_targeting


def test_tmlvts_basic():
    """Test basic functionality."""
    ic = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = tmle_var_targeting(ic)
    assert isinstance(result, dict)
    assert "sigma2" in result


def test_tmlvts_edge():
    """Test edge cases."""
    ic = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = tmle_var_targeting(ic)
    assert isinstance(result, dict)
