"""Tests for g_weight_simple.g_weight_simple."""

import math

from morie.fn import _array_core as np

from morie.fn.g_weight_simple import (
    g_weight_simple,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e17_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x_sample = rng.normal(0, 1, 100)
    xbar_sample = np.mean(x_sample)
    xbar_true = xbar_sample + 0.1
    s2_x = np.sum((x_sample - xbar_sample) ** 2) / (len(x_sample) - 1)
    x_k = x_sample[0]
    result = g_weight_simple(x_k, xbar_true, xbar_sample, s2_x)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e17_edge():
    """Test edge cases."""
    # When the sample mean equals the true mean, the g-weight reduces to 1
    rng = np.random.default_rng(42)
    x_sample = rng.normal(0, 1, 100)
    xbar_sample = np.mean(x_sample)
    xbar_true = xbar_sample
    s2_x = 1.0
    x_k = xbar_sample
    result = g_weight_simple(x_k, xbar_true, xbar_sample, s2_x)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isclose(result["value"], 1.0)
