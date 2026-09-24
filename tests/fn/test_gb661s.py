"""Tests for gb661s.gibbons_mw_sampsize."""

from morie.fn.gb661s import gibbons_mw_sampsize


def test_gb661s_basic():
    """Test basic functionality."""
    alpha = 0.05
    beta = 0.8
    delta = 0.5
    result = gibbons_mw_sampsize(alpha, beta, delta)
    assert isinstance(result, dict)
    assert "n" in result


def test_gb661s_edge():
    """Test edge cases."""
    alpha = 0.01
    beta = 0.9
    delta = 0.3
    result = gibbons_mw_sampsize(alpha, beta, delta)
    assert isinstance(result, dict)
