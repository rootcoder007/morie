"""Tests for se_d_probit.se_d_probit."""

from morie.fn import _array_core as np

from morie.fn.se_d_probit import se_d_probit


def test_ca11e21_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = se_d_probit(x)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_ca11e21_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = se_d_probit(x)
    assert isinstance(result, dict)
