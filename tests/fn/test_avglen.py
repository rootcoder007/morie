"""Tests for avglen.avg_path_length."""

from morie.fn import _array_core as np

from morie.fn.avglen import avg_path_length


def test_avglen_basic():
    """Test basic functionality."""
    G = np.eye(10)
    result = avg_path_length(G)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_avglen_edge():
    """Test edge cases."""
    G = np.eye(10)
    result = avg_path_length(G)
    assert isinstance(result, dict)
