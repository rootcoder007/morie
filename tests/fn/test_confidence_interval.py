"""Tests for confidence_interval.confidence_interval."""

import math

from morie.fn import _array_core as np

from morie.fn.confidence_interval import (
    confidence_interval,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r3e15_basic():
    """Test basic functionality."""
    estimate = 5.0
    variance = 2.0
    u_crit = 1.96
    result = confidence_interval(estimate, variance, u_crit)
    assert isinstance(result, dict)
    assert "lower" in result
    assert "value" in result
    assert "method" in result
    assert math.isfinite(result["lower"])


def test_the_r_series_dick_j_brus_spatial_sampling_with_r3e15_edge():
    """Test edge cases."""
    estimate = 0.0
    variance = 1.0
    u_crit = 2.576
    result = confidence_interval(estimate, variance, u_crit)
    assert isinstance(result, dict)
    assert "lower" in result
    assert "value" in result
    assert math.isfinite(result["lower"])
