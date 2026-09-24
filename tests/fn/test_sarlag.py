"""Tests for sarlag.spatial_ar_lag_model."""

from morie.fn import _array_core as np

from morie.fn.sarlag import spatial_ar_lag_model


def _make_W(n, rng, threshold=0.15):
    """Build a symmetric, row-normalized 0/1 spatial weights matrix."""
    raw = [[rng.uniform(0, 1) for _ in range(n)] for _ in range(n)]
    W = []
    for i in range(n):
        row = []
        for j in range(n):
            sym = (raw[i][j] + raw[j][i]) / 2
            row.append(1.0 if (i != j and sym < threshold) else 0.0)
        W.append(row)
    for i in range(n):
        s = sum(W[i])
        if s > 0:
            W[i] = [v / s for v in W[i]]
    return W


def test_sarlag_basic():
    """Test basic functionality."""
    n, p = 40, 3
    y = np.random.default_rng(43).normal(0, 1, n)
    X = np.random.default_rng(42).normal(0, 1, (n, p))
    W = _make_W(n, np.random.default_rng(41))
    result = spatial_ar_lag_model(y, X, W)
    assert isinstance(result, dict)


def test_sarlag_edge():
    """Test edge cases."""
    n, p = 20, 2
    y = np.random.default_rng(43).normal(0, 1, n)
    X = np.random.default_rng(42).normal(0, 1, (n, p))
    W = _make_W(n, np.random.default_rng(40), threshold=0.25)
    result = spatial_ar_lag_model(y, X, W)
    assert isinstance(result, dict)
