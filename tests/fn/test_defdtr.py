"""Tests for defdtr.deformable_detr."""

from morie.fn import _array_core as np

from morie.fn.defdtr import deformable_detr


def test_defdtr_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # x is an H x W feature map
    x = rng.normal(0, 1, (8, 10))
    # queries is a Q x 2 matrix of reference points in normalised [0, 1] coords
    queries = rng.uniform(0.0, 1.0, (5, 2))
    K = 4
    result = deformable_detr(x, queries, K)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "out" in result
    assert "samples" in result
    assert "ref_pixels" in result
    assert "Q" in result
    assert "K" in result
    assert result["Q"] == 5
    assert result["K"] == 4


def test_defdtr_edge():
    """Test edge cases."""
    rng = np.random.default_rng(7)
    # Smallest valid H x W feature map and Q x 2 queries, with K = 1
    x = rng.normal(0, 1, (3, 3))
    queries = rng.uniform(0.0, 1.0, (2, 2))
    K = 1
    result = deformable_detr(x, queries, K)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "out" in result
    assert "samples" in result
    assert "ref_pixels" in result
    assert result["Q"] == 2
    assert result["K"] == 1
