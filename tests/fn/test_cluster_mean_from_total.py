"""Tests for cluster_mean_from_total.cluster_mean_from_total."""

import math

from morie.fn import _array_core as np

from morie.fn.cluster_mean_from_total import (
    cluster_mean_from_total,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r6e10_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # t_hat: estimated cluster-sample total; m_population: number of clusters in M.
    t_hat = float(np.sum(rng.normal(10.0, 2.0, 25)))
    m_population = 50
    result = cluster_mean_from_total(t_hat, m_population)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])


def test_the_r_series_dick_j_brus_spatial_sampling_with_r6e10_edge():
    """Test edge cases."""
    # Minimal valid input: a single cluster in the population.
    t_hat = 7.5
    m_population = 1
    result = cluster_mean_from_total(t_hat, m_population)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
