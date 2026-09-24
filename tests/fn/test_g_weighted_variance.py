"""Tests for g_weighted_variance.g_weighted_variance."""

import math

from morie.fn import _array_core as np

from morie.fn.g_weighted_variance import (
    g_weighted_variance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e18_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    n_population = 200
    g = rng.uniform(0, 1, n)
    e = rng.normal(0, 1, n)
    result = g_weighted_variance(g, e, n, n_population)
    assert isinstance(result, dict)
    assert "value" in result
    value = result["value"]
    assert math.isfinite(value)
    assert value >= 0


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e18_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    n = 10
    n_population = 50
    g = rng.uniform(0.5, 1.5, n)
    e = rng.normal(0, 0.5, n)
    result = g_weighted_variance(g, e, n, n_population)
    assert isinstance(result, dict)
    assert "value" in result
    value = result["value"]
    assert math.isfinite(value)
    assert value >= 0
