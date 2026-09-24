"""Tests for gb_ksn.gibbons_ks_sample_size."""

from morie.fn.gb_ksn import gibbons_ks_sample_size


def test_gb_ksn_basic():
    """Test basic functionality."""
    epsilon = 0.1
    alpha = 0.05
    result = gibbons_ks_sample_size(epsilon, alpha)
    assert isinstance(result, dict)
    assert "n" in result
    assert result["n"] > 0


def test_gb_ksn_edge():
    """Test edge cases."""
    epsilon = 0.1
    alpha = 0.01
    result = gibbons_ks_sample_size(epsilon, alpha)
    assert isinstance(result, dict)
    assert "n" in result
    assert result["n"] > 0
