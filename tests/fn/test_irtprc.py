"""Tests for irtprc.partial_credit."""

from morie.fn import _array_core as np

from morie.fn.irtprc import partial_credit


def test_irtprc_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    theta = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    delta_j = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = partial_credit(y, theta, delta_j)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_irtprc_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    theta = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    delta_j = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = partial_credit(y, theta, delta_j)
    assert isinstance(result, dict)
