"""Tests for morloc.local_morans_i."""

from morie.fn import _array_core as np

from morie.fn.morloc import local_morans_i


def _make_weights(n, bandwidth=2):
    """Create a symmetric banded 0/1 spatial weight matrix."""
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and abs(i - j) <= bandwidth:
                W[i][j] = 1
    return W


def test_morloc_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    x = rng.normal(0, 1, n)
    W = _make_weights(n)
    result = local_morans_i(x, W)
    assert isinstance(result, dict)
    # Local Moran's I should return arrays of length n
    checked = False
    for value in result.values():
        if hasattr(value, '__len__') and not isinstance(value, str):
            assert len(value) == n
            checked = True
            break
    assert checked


def test_morloc_edge():
    """Test edge cases."""
    rng = np.random.default_rng(123)
    n = 40
    x = rng.normal(0, 1, n)
    W = _make_weights(n)
    result = local_morans_i(x, W)
    assert isinstance(result, dict)
