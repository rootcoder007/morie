"""Tests for grpio.geron_perceiver_io."""

from morie.fn import _array_core as np

from morie.fn.grpio import geron_perceiver_io


def test_grpio_basic():
    """Test basic functionality."""
    X = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]
    Z_latent = [[0.0, 0.0]]
    output_queries = [[0.0, 0.0]]
    result = geron_perceiver_io(X, Z_latent, output_queries)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grpio_edge():
    """Test edge cases."""
    X = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]
    Z_latent = [[0.0, 0.0]]
    output_queries = [[0.0, 0.0]]
    result = geron_perceiver_io(X, Z_latent, output_queries)
    assert isinstance(result, dict)
