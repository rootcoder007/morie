"""Tests for bcdblk.block_coordinate_descent."""

import math

from morie.fn import _array_core as np

from morie.fn.bcdblk import block_coordinate_descent


def _spd_matrix(rng, p):
    A = rng.normal(0, 1, (p, p))
    return [[sum(A[k][i] * A[k][j] for k in range(p)) + (1.0 if i == j else 0.0)
             for j in range(p)] for i in range(p)]


def test_bcdblk_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    p = 4
    Q = _spd_matrix(rng, p)
    b = rng.normal(0, 1, p)
    blocks = [[0, 1], [2, 3]]
    x0 = rng.normal(0, 1, p)
    result = block_coordinate_descent(Q, b, blocks, x0)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "x" in result
    assert "obj_trace" in result
    assert "n_iter" in result
    assert len(result["x"]) == p
    assert math.isfinite(result["estimate"])
    assert result["n_iter"] == 20
    assert len(result["obj_trace"]) == 21


def test_bcdblk_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    p = 3
    Q = _spd_matrix(rng, p)
    b = rng.normal(0, 1, p)
    blocks = [[0, 1, 2]]
    x0 = rng.normal(0, 1, p)
    result = block_coordinate_descent(Q, b, blocks, x0, n_iter=5)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "x" in result
    assert "n_iter" in result
    assert result["n_iter"] == 5
    assert len(result["x"]) == p
    assert math.isfinite(result["estimate"])
    assert len(result["obj_trace"]) == 6
