"""Tests for grvots.geron_soft_voting."""

import pytest

from morie.fn import _array_core as np
from morie.fn.grvots import geron_soft_voting


def _build_probs_3d(L, m, K, rng):
    """Build a shape-(L, m, K) array whose last axis sums to 1."""
    raw = rng.uniform(0.0, 1.0, (L, m, K))
    out = []
    for i in range(L):
        rows_i = []
        for j in range(m):
            vals = [float(raw[i][j][k]) for k in range(K)]
            s = sum(vals)
            rows_i.append([v / s for v in vals])
        out.append(rows_i)
    return out


def _build_probs_2d(L, K, rng):
    """Build a shape-(L, K) array whose rows sum to 1."""
    raw = rng.uniform(0.0, 1.0, (L, K))
    out = []
    for i in range(L):
        vals = [float(raw[i][k]) for k in range(K)]
        s = sum(vals)
        out.append([v / s for v in vals])
    return out


def test_grvots_basic():
    """Soft voting with several classifiers and several samples."""
    rng = np.random.default_rng(42)
    L, m, K = 3, 5, 4
    P = _build_probs_3d(L, m, K, rng)

    result = geron_soft_voting(P)

    assert isinstance(result, dict)
    assert "y_hat" in result
    assert "mean_probabilities" in result
    assert "margin" in result
    assert "confidence" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result

    assert len(result["y_hat"]) == m
    assert len(result["mean_probabilities"]) == m
    assert len(result["margin"]) == m
    for row in result["mean_probabilities"]:
        assert len(row) == K

    for yh in result["y_hat"]:
        assert 0 <= yh < K


def test_grvots_edge():
    """2-D input (L, K) with explicit non-uniform weights."""
    rng = np.random.default_rng(123)
    L, K = 4, 3
    P = _build_probs_2d(L, K, rng)
    weights = [1.0, 2.0, 3.0, 4.0]

    result = geron_soft_voting(P, weights=weights)

    assert isinstance(result, dict)
    assert "y_hat" in result
    assert "mean_probabilities" in result
    assert "margin" in result
    assert "confidence" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result

    # Single sample per classifier in the 2-D case.
    assert len(result["y_hat"]) == 1
    assert len(result["mean_probabilities"]) == 1
    assert len(result["mean_probabilities"][0]) == K
    assert 0 <= result["y_hat"][0] < K

    # Weighted average of valid pmf rows is itself a valid pmf.
    s = sum(result["mean_probabilities"][0])
    assert abs(s - 1.0) < 1e-6
