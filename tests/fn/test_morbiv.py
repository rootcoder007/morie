"""Tests for morbiv.bivariate_morans_i."""

import math

from morie.fn import _array_core as np
from morie.fn.morbiv import bivariate_morans_i


def _symmetric_W(n, seed):
    """Build a symmetric 0/1 adjacency matrix of shape (n, n) with zero diagonal."""
    rng = np.random.default_rng(seed)
    W = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            v = int(rng.integers(0, 2))
            W[i][j] = v
            W[j][i] = v
    return W


def test_morbiv_basic():
    """Test basic functionality."""
    n = 40
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    W = np.array(_symmetric_W(n, 44))
    result = bivariate_morans_i(x, y, W)
    assert isinstance(result, dict)
    stat_key = next(
        (k for k in result if k in ("I", "statistic", "estimate", "morans_i", "value")),
        None,
    )
    assert stat_key is not None
    assert math.isfinite(float(result[stat_key]))


def test_morbiv_edge():
    """Test edge cases with a smaller, sparse weight matrix."""
    n = 20
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    W = np.array(_symmetric_W(n, 11))
    result = bivariate_morans_i(x, y, W)
    assert isinstance(result, dict)
    assert len(result) > 0
    numeric_keys = [k for k, v in result.items() if isinstance(v, (int, float))]
    assert len(numeric_keys) >= 1
    assert all(math.isfinite(float(result[k])) for k in numeric_keys)
