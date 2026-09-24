"""Tests for snr2.snijders_bosker_r2_level1."""

from morie.fn import _array_core as np

from morie.fn.snr2 import snijders_bosker_r2_level1


def test_snr2_basic():
    """Test basic functionality."""
    sigma2_e1 = 0.5
    sigma2_u1 = 0.5
    sigma2_e0 = 0.5
    sigma2_u0 = 0.5
    result = snijders_bosker_r2_level1(sigma2_e1, sigma2_u1, sigma2_e0, sigma2_u0)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_snr2_edge():
    """Test edge cases."""
    sigma2_e1 = 0.5
    sigma2_u1 = 0.5
    sigma2_e0 = 0.5
    sigma2_u0 = 0.5
    result = snijders_bosker_r2_level1(sigma2_e1, sigma2_u1, sigma2_e0, sigma2_u0)
    assert isinstance(result, dict)
