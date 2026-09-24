"""Tests for moranI.morans_i_asymptotic_test."""

import math

from morie.fn import _array_core as np

from morie.fn.moranI import morans_i_asymptotic_test


def _build_symmetric_weights(n, p_fill=0.2, seed=0):
    """Build a symmetric 0/1 weight matrix with no self-loops."""
    rng = np.random.default_rng(seed)
    W = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rng.uniform(0, 1) < p_fill:
                W[i][j] = 1
                W[j][i] = 1
    return W


def test_moranI_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    x = rng.normal(0, 1, n)
    W = _build_symmetric_weights(n, p_fill=0.2, seed=42)
    result = morans_i_asymptotic_test(x, W)
    assert isinstance(result, dict)
    assert math.isfinite(result["p_value"])
    assert 0.0 <= result["p_value"] <= 1.0
    assert math.isfinite(result["estimate"])


def test_moranI_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    n = 40
    x = rng.normal(0, 1, n)
    W = _build_symmetric_weights(n, p_fill=0.2, seed=43)
    result = morans_i_asymptotic_test(x, W, alternative="less")
    assert isinstance(result, dict)
    assert math.isfinite(result["p_value"])
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["method"]
