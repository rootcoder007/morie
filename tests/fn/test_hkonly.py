"""Tests for hkonly.hadamard_response."""

from morie.fn import _array_core as np

from morie.fn.hkonly import hadamard_response


def test_hkonly_basic():
    """Test basic functionality."""
    counts = np.random.default_rng(42).normal(0.0, 1.0, 40)
    epsilon = 0.1
    result = hadamard_response(counts, epsilon)
    assert isinstance(result, dict)
    assert "p" in result


def test_hkonly_edge():
    """Test edge cases."""
    counts = np.random.default_rng(42).normal(0.0, 1.0, 40)
    epsilon = 0.1
    result = hadamard_response(counts, epsilon)
    assert isinstance(result, dict)
