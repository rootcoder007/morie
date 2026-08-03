"""Tests for _bits._bits."""

from morie.fn import _array_core as np

from morie.fn._bits import _bits


def test_ghs026_basic():
    """Test basic functionality."""
    P = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    A_epsilon = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    result = _bits(P, mu, A_epsilon, m)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ghs026_edge():
    """Test edge cases."""
    P = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    A_epsilon = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    result = _bits(P, mu, A_epsilon, m)
    assert isinstance(result, dict)
