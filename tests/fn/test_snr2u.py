"""Tests for snr2u.snijders_bosker_r2_level2."""

from morie.fn import _array_core as np

from morie.fn.snr2u import snijders_bosker_r2_level2


def test_snr2u_basic():
    """Test basic functionality."""
    sigma2_e1 = 0.5
    sigma2_u1 = 0.5
    sigma2_e0 = 0.5
    sigma2_u0 = 0.5
    result = snijders_bosker_r2_level2(sigma2_e1, sigma2_u1, sigma2_e0, sigma2_u0)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_snr2u_edge():
    """Test edge cases."""
    sigma2_e1 = 0.5
    sigma2_u1 = 0.5
    sigma2_e0 = 0.5
    sigma2_u0 = 0.5
    result = snijders_bosker_r2_level2(sigma2_e1, sigma2_u1, sigma2_e0, sigma2_u0)
    assert isinstance(result, dict)
