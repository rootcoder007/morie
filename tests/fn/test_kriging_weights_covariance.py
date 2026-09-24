"""Tests for kriging_weights_covariance.kriging_weights_covariance."""

import math

from morie.fn import _array_core as np
from morie.fn.kriging_weights_covariance import (
    kriging_weights_covariance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r21e4_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 5
    A = rng.normal(0, 1, (n, n))
    # Symmetric positive semi-definite covariance matrix for samples (A A^T)
    cov_ss = [
        [sum(A[i][k] * A[j][k] for k in range(n)) for j in range(n)]
        for i in range(n)
    ]
    cov_s0 = rng.normal(0, 1, n)
    result = kriging_weights_covariance(cov_ss, cov_s0)
    assert isinstance(result, dict)
    assert "nu" in result
    assert "lam" in result
    assert "value" in result
    assert "method" in result
    assert math.isfinite(result["nu"])
    assert math.isfinite(result["value"])


def test_the_r_series_dick_j_brus_spatial_sampling_with_r21e4_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    n = 3
    A = rng.normal(0, 1, (n, n))
    cov_ss = [
        [sum(A[i][k] * A[j][k] for k in range(n)) for j in range(n)]
        for i in range(n)
    ]
    cov_s0 = rng.normal(0, 1, n)
    result = kriging_weights_covariance(cov_ss, cov_s0)
    assert isinstance(result, dict)
    assert "nu" in result
    assert "lam" in result
    assert "method" in result
    assert len(result["lam"]) == n
    assert math.isfinite(result["nu"])
