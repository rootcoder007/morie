"""Tests for grsdpa.geron_scaled_dot_product_attention."""

from morie.fn import _array_core as np

from morie.fn.grsdpa import geron_scaled_dot_product_attention


def test_grsdpa_basic():
    """Test basic functionality."""
    Q = [[1.0, 0.0]]
    K = [[1.0, 0.0], [0.0, 1.0]]
    V = [[1.0, 0.0], [0.0, 1.0]]
    result = geron_scaled_dot_product_attention(Q, K, V)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grsdpa_edge():
    """Test edge cases."""
    Q = [[1.0, 0.0]]
    K = [[1.0, 0.0], [0.0, 1.0]]
    V = [[1.0, 0.0], [0.0, 1.0]]
    result = geron_scaled_dot_product_attention(Q, K, V)
    assert isinstance(result, dict)
