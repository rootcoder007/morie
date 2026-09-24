"""Tests for sarerr.spatial_ar_error_model."""

from morie.fn import _array_core as np

from morie.fn.sarerr import spatial_ar_error_model


def _grid_W(n):
    """Build a row-standardised 1-D contiguity weights matrix."""
    rows = []
    for i in range(n):
        row = [0] * n
        if i - 1 >= 0:
            row[i - 1] = 1
        if i + 1 < n:
            row[i + 1] = 1
        s = sum(row)
        if s > 0:
            row = [v / s for v in row]
        rows.append(row)
    return rows


def test_sarerr_basic():
    """Test basic functionality."""
    n, p = 40, 3
    y = np.random.default_rng(43).normal(0, 1, n)
    X = np.random.default_rng(42).normal(0, 1, (n, p))
    W = _grid_W(n)
    result = spatial_ar_error_model(y, X, W)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_sarerr_edge():
    """Test edge cases."""
    n, p = 10, 2
    y = np.random.default_rng(43).normal(0, 1, n)
    X = np.random.default_rng(42).normal(0, 1, (n, p))
    W = _grid_W(n)
    result = spatial_ar_error_model(y, X, W)
    assert isinstance(result, dict)
    assert len(result) > 0
