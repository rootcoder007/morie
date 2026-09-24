"""Tests for beta_posterior_interval_prob.beta_posterior_interval_prob."""

import math

from morie.fn import _array_core as np

from morie.fn.beta_posterior_interval_prob import (
    beta_posterior_interval_prob,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r12e18_basic():
    """Test basic functionality."""
    z = 30          # number of successes
    n = 100         # number of trials
    c = 1           # prior shape parameter
    d = 1           # prior shape parameter
    v = 0.2         # lower bound of the interval
    l = 0.1         # length of the interval
    result = beta_posterior_interval_prob(v, l, z, n, c, d)
    assert isinstance(result, dict)
    assert "value" in result
    value = result["value"]
    assert math.isfinite(value)
    assert 0.0 <= value <= 1.0


def test_the_r_series_dick_j_brus_spatial_sampling_with_r12e18_edge():
    """Test edge cases."""
    z = 5
    n = 20
    c = 2
    d = 2
    v = 0.0
    l = 1.0
    result = beta_posterior_interval_prob(v, l, z, n, c, d)
    assert isinstance(result, dict)
    assert "value" in result
    value = result["value"]
    assert math.isfinite(value)
    assert 0.0 <= value <= 1.0
