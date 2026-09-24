"""Tests for gb_ssj.gibbons_sign_sample_size_2."""

from morie.fn.gb_ssj import gibbons_sign_sample_size_2


def test_gb_ssj_basic():
    """Test basic functionality."""
    alpha = 0.05
    beta = 0.2
    p0 = 0.5
    result = gibbons_sign_sample_size_2(alpha, beta, p0)
    assert isinstance(result, dict)
    assert "n" in result
    assert isinstance(result["n"], int)
    assert result["n"] > 0


def test_gb_ssj_edge():
    """Test edge cases."""
    alpha = 0.05
    beta = 0.1
    p0 = 0.6
    result = gibbons_sign_sample_size_2(alpha, beta, p0)
    assert isinstance(result, dict)
    assert "n" in result
    assert isinstance(result["n"], int)
    assert result["n"] > 0
