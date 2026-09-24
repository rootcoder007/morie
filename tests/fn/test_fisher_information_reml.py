"""Tests for fisher_information_reml.fisher_information_reml."""

import math

from morie.fn import _array_core as np

from morie.fn.fisher_information_reml import (
    fisher_information_reml,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r24e2_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 5
    # Covariance matrix A: identity is a valid positive-definite covariance
    a = np.eye(n)
    # Derivative matrices dA_i with respect to two covariance parameters
    da_list = [np.eye(n), np.eye(n)]
    result = fisher_information_reml(a, da_list)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(float(result["value"]))


def test_the_r_series_dick_j_brus_spatial_sampling_with_r24e2_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 3
    # Smallest reasonable positive-definite covariance matrix
    a = np.eye(n)
    # Single derivative (single covariance parameter)
    da_list = [np.eye(n)]
    result = fisher_information_reml(a, da_list)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(float(result["value"]))
