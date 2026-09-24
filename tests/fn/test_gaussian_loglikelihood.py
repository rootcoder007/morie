"""Tests for gaussian_loglikelihood.gaussian_loglikelihood."""

import math

from morie.fn import _array_core as np

from morie.fn.gaussian_loglikelihood import (
    gaussian_loglikelihood,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r21e23_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 10
    z = rng.normal(0, 1, n)
    mu = rng.normal(0, 1, n)
    cov = np.eye(n)
    result = gaussian_loglikelihood(z, mu, cov)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])


def test_the_r_series_dick_j_brus_spatial_sampling_with_r21e23_edge():
    """Test edge case: z coincides with mu so the quadratic form vanishes."""
    rng = np.random.default_rng(0)
    n = 5
    mu = rng.normal(0, 1, n)
    z = mu
    cov = np.eye(n)
    result = gaussian_loglikelihood(z, mu, cov)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
