"""Tests for expected_squared_distance.expected_squared_distance."""

import math

from morie.fn import _array_core as np

from morie.fn.expected_squared_distance import (
    expected_squared_distance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r13e15_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    zhat_i = float(rng.normal(0, 1))
    zhat_j = float(rng.normal(0, 1))
    r2 = float(rng.uniform(0.5, 5.0))
    s2_i = float(rng.uniform(0.1, 2.0))
    s2_j = float(rng.uniform(0.1, 2.0))
    s2_ij = float(rng.uniform(0.0, 1.0))
    result = expected_squared_distance(zhat_i, zhat_j, r2, s2_i, s2_j, s2_ij)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])


def test_the_r_series_dick_j_brus_spatial_sampling_with_r13e15_edge():
    """Test edge cases."""
    zhat_i = 0.0
    zhat_j = 0.0
    r2 = 1.0
    s2_i = 0.0
    s2_j = 0.0
    s2_ij = 0.0
    result = expected_squared_distance(zhat_i, zhat_j, r2, s2_i, s2_j, s2_ij)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
