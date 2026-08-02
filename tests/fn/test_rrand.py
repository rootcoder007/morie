"""Tests for rrand.randomized_response."""

from morie.fn import _array_core as np

from morie.fn.rrand import randomized_response


def test_rrand_basic():
    """Test basic functionality."""
    bit = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = randomized_response(bit, epsilon)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_rrand_edge():
    """Test edge cases."""
    bit = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = randomized_response(bit, epsilon)
    assert isinstance(result, dict)
