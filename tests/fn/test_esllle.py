"""Tests for esllle.esl_lle."""

import math

import pytest

from morie.fn.esllle import esl_lle


def _spiral(n):
    """A one-dimensional spiral embedded in the plane."""
    pts = []
    for i in range(n):
        t = 0.5 + 4.0 * math.pi * i / (n - 1)
        pts.append([t * math.cos(t), t * math.sin(t)])
    return pts


def test_esllle_basic():
    """Reconstruction weights sum to one and the embedding has k columns."""
    n = 40
    X = _spiral(n)
    k, m = 2, 6
    result = esl_lle(X, k=k, neighbors=m)

    assert result["neighbors"] == m
    W = result["weights"]
    assert W.shape == (n, n)
    # The sum-to-one constraint is what buys the affine invariance.
    for i in range(n):
        assert abs(sum(float(v) for v in W[i]) - 1.0) < 1e-9
        # Only the m neighbours carry weight, and never the point itself.
        assert float(W[i][i]) == 0.0
        assert sum(1 for v in W[i] if float(v) != 0.0) <= m

    emb = result["embedding"]
    assert emb.shape == (n, k)
    # The constant bottom eigenvector is dropped, so no column is degenerate.
    for j in range(k):
        col = [float(emb[i][j]) for i in range(n)]
        mean = sum(col) / n
        var = sum((c - mean) ** 2 for c in col) / n
        assert math.sqrt(var) > 1e-8

    assert result["reconstruction_error"] >= 0.0
    # Eigenvalues of (I-W)^T (I-W) are non-negative and returned in order.
    ev = [float(v) for v in result["eigenvalues"]]
    assert len(ev) == k + 1
    assert ev[0] < 1e-8            # the discarded constant eigenvector
    assert all(ev[i] <= ev[i + 1] + 1e-12 for i in range(k))
    assert all(v > -1e-9 for v in ev)


def test_esllle_edge():
    """Weights reproduce a point exactly when it lies in its neighbours hull."""
    # Collinear, evenly spaced points: the interior point is the midpoint of
    # its two neighbours, so the fitted weights are 1/2 and 1/2 and the
    # reconstruction is exact up to the ridge on the Gram matrix.
    X = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]
    result = esl_lle(X, k=1, neighbors=2, reg=1e-12)
    W = result["weights"]
    assert abs(float(W[2][1]) - 0.5) < 1e-6
    assert abs(float(W[2][3]) - 0.5) < 1e-6
    assert result["reconstruction_error"] < 1e-6

    with pytest.raises(ValueError, match="k must be between"):
        esl_lle(X, k=0, neighbors=2)
    with pytest.raises(ValueError, match="neighbors must be between"):
        esl_lle(X, k=1, neighbors=5)
