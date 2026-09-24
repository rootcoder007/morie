"""Tests for expected_stratum_variance.expected_stratum_variance."""

import math

from morie.fn import _array_core as np

from morie.fn.expected_stratum_variance import (
    expected_stratum_variance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r13e16_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n_h = 40
    # Sample points in p=3 dimensions; compute the upper-triangular sum of
    # squared pairwise distances d2_ij (i<j), which plays the role of
    # d2_upper_sum in the formula.
    pts = rng.normal(0, 1, (n_h, 3))
    d2_upper_sum = 0.0
    for i in range(n_h):
        for j in range(i + 1, n_h):
            diff = pts[i] - pts[j]
            d2_upper_sum += sum(d * d for d in diff)

    result = expected_stratum_variance(d2_upper_sum, n_h)

    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert result["value"] >= 0.0
    assert result["method"] == "Brus (2022) eq. (13.16)"


def test_the_r_series_dick_j_brus_spatial_sampling_with_r13e16_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n_h = 3
    # Minimal valid stratum: a handful of points in 2-D.
    pts = rng.normal(0, 1, (n_h, 2))
    d2_upper_sum = 0.0
    for i in range(n_h):
        for j in range(i + 1, n_h):
            diff = pts[i] - pts[j]
            d2_upper_sum += sum(d * d for d in diff)

    result = expected_stratum_variance(d2_upper_sum, n_h)

    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert result["value"] >= 0.0
    assert result["method"] == "Brus (2022) eq. (13.16)"
