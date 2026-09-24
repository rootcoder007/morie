"""Tests for expected_tau2.expected_tau2."""

import math

from morie.fn import _array_core as np

from morie.fn.expected_tau2 import (
    expected_tau2,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r24e5_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    k, m = 3, 2
    # cov_theta: k x k symmetric PSD via raw @ raw.T
    raw = rng.normal(0, 1, (k, k))
    cov_theta = [
        [sum(raw[i][t] * raw[j][t] for t in range(k)) for j in range(k)]
        for i in range(k)
    ]
    # dlam_dtheta: k x m matrix of partial derivatives
    dlam_dtheta = rng.normal(0, 1, (k, m))
    # a: m x m symmetric PSD via raw_a @ raw_a.T
    raw_a = rng.normal(0, 1, (m, m))
    a = [
        [sum(raw_a[i][t] * raw_a[j][t] for t in range(m)) for j in range(m)]
        for i in range(m)
    ]

    result = expected_tau2(cov_theta, dlam_dtheta, a)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])


def test_the_r_series_dick_j_brus_spatial_sampling_with_r24e5_edge():
    """Test edge cases - identity covariance/design matrices."""
    rng = np.random.default_rng(43)
    k, m = 2, 2
    cov_theta = np.eye(k)
    dlam_dtheta = rng.normal(0, 1, (k, m))
    a = np.eye(m)

    result = expected_tau2(cov_theta, dlam_dtheta, a)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
