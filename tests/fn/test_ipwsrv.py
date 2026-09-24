"""Tests for ipwsrv.ipw_with_survey_weights."""

import math

from morie.fn import _array_core as np

from morie.fn.ipwsrv import ipw_with_survey_weights


def test_ipwsrv_basic():
    """Test basic functionality with survey weights and propensity scores."""
    rng = np.random.default_rng(43)
    n = 100
    y = rng.normal(0, 1, n)
    T = rng.integers(0, 2, n)
    weights = rng.uniform(0.5, 2.0, n)
    propensity = rng.uniform(0.05, 0.95, n)
    result = ipw_with_survey_weights(y, T, weights, propensity)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert result["n"] == n
    assert math.isfinite(result["se"])
    assert math.isfinite(result["mu1"])
    assert math.isfinite(result["mu0"])
    assert result["ess"] > 0


def test_ipwsrv_edge():
    """Test edge case: weights=None falls back to unit design weights."""
    rng = np.random.default_rng(43)
    n = 40
    y = rng.normal(0, 1, n)
    T = rng.integers(0, 2, n)
    propensity = rng.uniform(0.1, 0.9, n)
    result = ipw_with_survey_weights(y, T, None, propensity)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert result["n"] == n
    assert result["sum_w"] > 0
    assert result["ess"] > 0
