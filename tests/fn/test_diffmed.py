"""Tests for diffmed.difference_in_coefficients."""

from morie.fn import _array_core as np

from morie.fn.diffmed import difference_in_coefficients


def test_diffmed_basic():
    """Test basic functionality."""
    c = 0.7
    c_prime = 0.5
    result = difference_in_coefficients(c, c_prime)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "indirect" in result
    assert result["indirect"] == c - c_prime
    assert result["estimate"] == c - c_prime


def test_diffmed_edge():
    """Test edge cases."""
    c = 0.7
    c_prime = 0.5
    result = difference_in_coefficients(c, c_prime)
    assert isinstance(result, dict)
    assert result["total"] == c
    assert result["direct"] == c_prime
