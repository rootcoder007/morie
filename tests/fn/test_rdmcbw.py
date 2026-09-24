"""Tests for rdmcbw.mse_optimal_bandwidth_rdd."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.rdmcbw import mse_optimal_bandwidth_rdd


def test_rdmcbw_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_x = np.random.default_rng(42)
    n = 200
    y = rng_y.normal(0, 1, n)
    x = rng_x.normal(0, 1, n)
    cutoff = 0.0
    result = mse_optimal_bandwidth_rdd(y, x, cutoff)
    assert isinstance(result, dict)
    for key in (
        "estimate",
        "h_opt",
        "h_no_reg",
        "h1",
        "f_hat",
        "sigma2",
        "m3",
        "h2_plus",
        "h2_minus",
        "m2_plus",
        "m2_minus",
        "r_plus",
        "r_minus",
        "n_plus",
        "n_minus",
        "n2_plus",
        "n2_minus",
        "n",
    ):
        assert key in result
    assert result["n"] == n
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["h_opt"])
    assert result["h_opt"] > 0.0
    assert result["h1"] > 0.0
    assert result["n_plus"] >= 2
    assert result["n_minus"] >= 2


def test_rdmcbw_edge():
    """Test edge cases with a uniform-kernel constant and an asymmetric cutoff."""
    rng_y = np.random.default_rng(43)
    rng_x = np.random.default_rng(42)
    n = 300
    y = rng_y.normal(0, 1, n)
    x = rng_x.uniform(-2.0, 2.0, n)
    cutoff = -0.5
    result = mse_optimal_bandwidth_rdd(y, x, cutoff, kernel_constant=5.4)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "h_opt" in result
    assert "h1" in result
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["h_opt"])
    assert result["h_opt"] > 0.0
    assert result["h1"] > 0.0
