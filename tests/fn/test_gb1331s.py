"""Tests for gb1331s.gibbons_sign_efficacy."""

from morie.fn import _array_core as np

from morie.fn.gb1331s import gibbons_sign_efficacy


def test_gb1331s_basic():
    """Test basic functionality."""
    N = 100
    fmed = 0.4
    result = gibbons_sign_efficacy(N, fmed)
    assert isinstance(result, dict)
    assert "efficacy" in result
    assert "per_obs" in result
    assert "n" in result
    assert "fmed" in result
    assert "method" in result
    # Independent computation of the documented formula
    expected_efficacy = 4.0 * N * fmed * fmed
    assert result["efficacy"] == expected_efficacy
    assert result["per_obs"] == expected_efficacy / N
    assert result["n"] == N
    assert result["fmed"] == fmed


def test_gb1331s_edge():
    """Test edge cases."""
    N = 1
    fmed = 0.1
    result = gibbons_sign_efficacy(N, fmed)
    assert isinstance(result, dict)
    assert result["efficacy"] == 4.0 * N * fmed * fmed
    assert result["per_obs"] == 4.0 * fmed * fmed
    assert result["n"] == N
    assert result["fmed"] == fmed
