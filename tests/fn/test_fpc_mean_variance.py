"""Tests for fpc_mean_variance.fpc_mean_variance."""

import math

from morie.fn import _array_core as np

from morie.fn.fpc_mean_variance import (
    fpc_mean_variance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r26e5_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    x = rng.normal(0, 1, n)
    mean = np.mean(x)
    s2 = np.sum([(xi - mean) ** 2 for xi in x]) / (n - 1)
    n_population = 1000
    result = fpc_mean_variance(s2, n, n_population)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])


def test_the_r_series_dick_j_brus_spatial_sampling_with_r26e5_edge():
    """Test edge case: n_population equals n, correction factor zero."""
    rng = np.random.default_rng(42)
    n = 20
    x = rng.normal(0, 1, n)
    mean = np.mean(x)
    s2 = np.sum([(xi - mean) ** 2 for xi in x]) / (n - 1)
    n_population = n
    result = fpc_mean_variance(s2, n, n_population)
    assert isinstance(result, dict)
    assert "value" in result
    assert result["value"] == 0.0
