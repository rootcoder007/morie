"""Tests for sarmix.spatial_ar_combined."""

import pytest

from morie.fn import _array_core as np

from morie.fn.sarmix import spatial_ar_combined


def _make_ring_weights(n):
    """Create a row-normalized ring adjacency weights matrix."""
    W = [[0.0] * n for _ in range(n)]
    for i in range(n):
        W[i][(i + 1) % n] = 1.0
        W[i][(i - 1) % n] = 1.0
        row_sum = sum(W[i])
        if row_sum > 0:
            W[i] = [x / row_sum for x in W[i]]
    return W


def test_sarmix_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    p = 3
    # Build design matrix with explicit intercept as first column
    X = []
    for _ in range(n):
        row = [1.0] + list(rng.normal(0, 1, p - 1))
        X.append(row)
    y = rng.normal(0, 1, n)
    W1 = _make_ring_weights(n)
    W2 = _make_ring_weights(n)
    result = spatial_ar_combined(y, X, W1, W2)
    # Assert that result contains all expected payload keys
    assert "estimate" in result
    assert "se" in result
    assert "rho" in result
    assert "lambda" in result
    assert "sigma2" in result
    assert "loglik" in result
    assert "n" in result
    assert "method" in result
    # Check that n matches
    assert result["n"] == n


def test_sarmix_edge():
    """Test edge case: shape mismatch raises ValueError."""
    rng = np.random.default_rng(42)
    n = 40
    p = 3
    X = []
    for _ in range(n):
        row = [1.0] + list(rng.normal(0, 1, p - 1))
        X.append(row)
    y = rng.normal(0, 1, n)
    W1 = _make_ring_weights(n)
    W2 = _make_ring_weights(n)
    # Create W1 with wrong shape (n, n-1)
    W1_wrong = [[0.0] * (n - 1) for _ in range(n)]
    with pytest.raises(ValueError):
        spatial_ar_combined(y, X, W1_wrong, W2)
