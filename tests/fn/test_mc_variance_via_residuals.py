"""Tests for mc_variance_via_residuals.mc_variance_via_residuals."""

import math

from morie.fn import _array_core as np

from morie.fn.mc_variance_via_residuals import (
    mc_variance_via_residuals,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e42_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    e = rng.normal(0, 1, 40)
    pi = rng.uniform(0.1, 0.5, 40)
    n_population = 200
    result = mc_variance_via_residuals(e, pi, n_population)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e42_edge():
    """Test edge cases with a small but valid sample."""
    rng = np.random.default_rng(123)
    e = rng.normal(0, 1, 5)
    pi = rng.uniform(0.2, 0.8, 5)
    n_population = 20
    result = mc_variance_via_residuals(e, pi, n_population)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
