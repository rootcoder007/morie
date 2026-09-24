"""Tests for difference_estimator.difference_estimator."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.difference_estimator import (
    difference_estimator,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e2_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n_population = 200
    n_sample = 40

    # True population values and model predictions for the entire population
    z_population = rng.normal(0, 1, n_population)
    m_all = rng.normal(0, 1, n_population)

    # Simple random sample without replacement
    sample_indices = rng.choice(n_population, n_sample, replace=False)
    z_sample = [z_population[i] for i in sample_indices]
    m_sample = [m_all[i] for i in sample_indices]
    pi_sample = [n_sample / n_population] * n_sample

    result = difference_estimator(m_all, z_sample, m_sample, pi_sample, n_population)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e2_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n_population = 50
    n_sample = 10

    # True population values and model predictions
    z_population = rng.normal(0, 1, n_population)
    m_all = rng.normal(0, 1, n_population)

    # Small sample
    sample_indices = rng.choice(n_population, n_sample, replace=False)
    z_sample = [z_population[i] for i in sample_indices]
    m_sample = [m_all[i] for i in sample_indices]
    pi_sample = [n_sample / n_population] * n_sample

    result = difference_estimator(m_all, z_sample, m_sample, pi_sample, n_population)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
