"""Tests for gb5714.gibbons_wsrt_sampsize."""

from morie.fn import _array_core as np

from morie.fn.gb5714 import gibbons_wsrt_sampsize


def test_gb5714_basic():
    """Test basic functionality."""
    alpha = 0.05
    beta = 0.8
    delta = 0.5
    sigma = 1.0
    result = gibbons_wsrt_sampsize(alpha, beta, delta, sigma)
    assert isinstance(result, dict)
    assert "n" in result


def test_gb5714_edge():
    """Test edge cases."""
    alpha = 0.01
    beta = 0.9
    delta = 0.2
    sigma = 1.0
    result = gibbons_wsrt_sampsize(alpha, beta, delta, sigma)
    assert isinstance(result, dict)
    assert "n" in result
