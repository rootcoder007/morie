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


def test_deformable_attention_recomputed():
    """Bilinear samples at reference point + offset, weighted average."""
    import math

    import pytest

    F = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]

    def bil(y, x):
        y = min(max(y, 0.0), 2.0)
        x = min(max(x, 0.0), 2.0)
        y0, x0 = int(math.floor(y)), int(math.floor(x))
        y1, x1 = min(y0 + 1, 2), min(x0 + 1, 2)
        dy, dx = y - y0, x - x0
        return (
            F[y0][x0] * (1 - dy) * (1 - dx)
            + F[y0][x1] * (1 - dy) * dx
            + F[y1][x0] * dy * (1 - dx)
            + F[y1][x1] * dy * dx
        )

    q = [[0.25, 0.5]]
    off = [0.3, -0.4, -0.2, 0.7]
    wt = [0.6, 0.4]
    ry, rx = 0.5 * 2, 0.25 * 2
    v = [bil(ry + off[0], rx + off[1]), bil(ry + off[2], rx + off[3])]
    r = deformable_detr(F, q, K=2, offsets=off, weights=wt)
    assert r["samples"][0] == pytest.approx(v, rel=1e-14)
    assert r["out"][0] == pytest.approx((0.6 * v[0] + 0.4 * v[1]) / 1.0, rel=1e-14)
