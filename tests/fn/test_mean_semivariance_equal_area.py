"""Tests for mean_semivariance_equal_area.mean_semivariance_equal_area."""

import math

from morie.fn import _array_core as np
from morie.fn.mean_semivariance_equal_area import (
    mean_semivariance_equal_area,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r13e7_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    # gamma_bar_h: average semivariance values at different lag distances (non-negative)
    gamma_bar_h = [abs(v) for v in rng.normal(0.0, 1.0, 10)]
    result = mean_semivariance_equal_area(gamma_bar_h, n)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert result["value"] >= 0.0
    assert "method" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r13e7_edge():
    """Test edge cases with a small valid input."""
    rng = np.random.default_rng(0)
    n = 5
    gamma_bar_h = [abs(v) for v in rng.normal(0.0, 1.0, 4)]
    result = mean_semivariance_equal_area(gamma_bar_h, n)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert result["value"] >= 0.0
