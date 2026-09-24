"""Tests for lismor.local_morans_i."""

import math

from morie.fn import _array_core as np

from morie.fn.lismor import local_morans_i


def test_lismor_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    x = rng.normal(0, 1, n)
    W = rng.uniform(0, 1, (n, n))
    # Set diagonal to zero (common for weight matrices)
    for i in range(n):
        W[i][i] = 0
    result = local_morans_i(x, W)
    assert isinstance(result, dict)
    assert len(result) > 0
    key = next(iter(result))
    values = result[key]
    assert isinstance(values, list)
    assert len(values) == n
    assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def test_lismor_edge():
    """Test edge case with small n."""
    rng = np.random.default_rng(42)
    n = 5
    x = rng.normal(0, 1, n)
    W = rng.uniform(0, 1, (n, n))
    for i in range(n):
        W[i][i] = 0
    result = local_morans_i(x, W)
    assert isinstance(result, dict)
    assert len(result) > 0
    key = next(iter(result))
    values = result[key]
    assert isinstance(values, list)
    assert len(values) == n
    assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
