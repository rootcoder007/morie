"""Tests for gls_estimator.gls_estimator."""

import math

from morie.fn import _array_core as np

from morie.fn.gls_estimator import (
    gls_estimator,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r15e10_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    x = rng.normal(0, 1, (n, p))
    # Build a symmetric positive-definite covariance matrix C = A^T A + I*n
    a = rng.normal(0, 1, (n, n))
    c = [[sum(a[k][i] * a[k][j] for k in range(n)) for j in range(n)]
         for i in range(n)]
    for i in range(n):
        c[i][i] += 1.0
    zhat = rng.normal(0, 1, n)
    result = gls_estimator(x, c, zhat)
    assert isinstance(result, dict)
    assert "value" in result
    assert isinstance(result["value"], (int, float))
    assert math.isfinite(result["value"])
    assert "values" in result
    assert "method" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r15e10_edge():
    """Test edge cases."""
    rng = np.random.default_rng(7)
    n, p = 8, 2
    x = rng.normal(0, 1, (n, p))
    a = rng.normal(0, 1, (n, n))
    c = [[sum(a[k][i] * a[k][j] for k in range(n)) for j in range(n)]
         for i in range(n)]
    for i in range(n):
        c[i][i] += 1.0
    zhat = rng.normal(0, 1, n)
    result = gls_estimator(x, c, zhat)
    assert isinstance(result, dict)
    assert "value" in result
    assert isinstance(result["value"], (int, float))
    assert math.isfinite(result["value"])
