"""Tests for otc2p.ot_cost_lp."""

from morie.fn import _array_core as np

from morie.fn.otc2p import ot_cost_lp


def test_otc2p_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = ot_cost_lp(X, Y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_otc2p_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = ot_cost_lp(X, Y)
    assert isinstance(result, dict)
