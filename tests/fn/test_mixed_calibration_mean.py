"""Tests for mixed_calibration_mean.mixed_calibration_mean."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.mixed_calibration_mean import (
    mixed_calibration_mean,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e36_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n_sample = 40
    n_population = 1000

    zbar_pi = float(np.mean(rng.normal(5.0, 1.0, n_sample)))
    a_hat = 0.5
    pi_sample = rng.uniform(0.05, 0.5, n_sample)
    m_all_mean = 10.0
    m_ht_mean = 10.2
    b_hat = 0.3

    result = mixed_calibration_mean(
        zbar_pi,
        a_hat,
        pi_sample,
        m_all_mean,
        m_ht_mean,
        b_hat,
        n_population,
    )
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e36_edge():
    """Test edge cases with a minimal valid input."""
    rng = np.random.default_rng(7)
    n_sample = 5
    n_population = 100

    zbar_pi = 0.0
    a_hat = 0.0
    pi_sample = rng.uniform(0.01, 0.1, n_sample)
    m_all_mean = 1.0
    m_ht_mean = 1.0
    b_hat = 0.0

    result = mixed_calibration_mean(
        zbar_pi,
        a_hat,
        pi_sample,
        m_all_mean,
        m_ht_mean,
        b_hat,
        n_population,
    )
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert result["value"] == 0.0
