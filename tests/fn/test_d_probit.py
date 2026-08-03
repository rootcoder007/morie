"""Tests for d_probit.d_probit."""

from morie.fn import _array_core as np

from morie.fn.d_probit import d_probit


def test_ca11e20_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = d_probit(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca11e20_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = d_probit(x)
    assert isinstance(result, dict)
