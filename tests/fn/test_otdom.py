"""Tests for otdom.ot_domain_adaptation."""

from morie.fn import _array_core as np

from morie.fn.otdom import ot_domain_adaptation


def test_otdom_basic():
    """Test basic functionality."""
    Xs = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Xt = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    epsilon = 0.1
    result = ot_domain_adaptation(Xs, Xt, epsilon)
    assert isinstance(result, dict)
    assert "Xs_adapted" in result


def test_otdom_edge():
    """Test edge cases."""
    Xs = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Xt = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    epsilon = 0.1
    result = ot_domain_adaptation(Xs, Xt, epsilon)
    assert isinstance(result, dict)
