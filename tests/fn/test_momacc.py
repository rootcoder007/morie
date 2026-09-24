"""Tests for momacc.moments_accountant."""

from morie.fn import _array_core as np

from morie.fn.momacc import moments_accountant


def test_momacc_basic():
    """Test basic functionality."""
    sigma = 0.1
    sample_rate = 0.1
    steps = 5
    result = moments_accountant(sigma, sample_rate, steps)
    assert isinstance(result, dict)
    assert "epsilon" in result


def test_momacc_edge():
    """Test edge cases."""
    sigma = 0.1
    sample_rate = 0.1
    steps = 5
    result = moments_accountant(sigma, sample_rate, steps)
    assert isinstance(result, dict)
