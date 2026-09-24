"""Tests for hmumap.geron_umap."""

from morie.fn import _array_core as np

from morie.fn.hmumap import geron_umap


def test_hmumap_basic():
    """Test basic functionality."""
    X = [[0.0], [1.0], [3.0], [7.0], [7.5], [12.0]]
    result = geron_umap(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmumap_edge():
    """Test edge cases."""
    X = [[0.0], [1.0], [3.0], [7.0], [7.5], [12.0]]
    result = geron_umap(X)
    assert isinstance(result, dict)
