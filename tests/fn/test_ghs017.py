"""Tests for ghs017.ghosal_ch3_discrete_random_measure."""

from morie.fn import _array_core as np

from morie.fn.ghs017 import ghosal_ch3_discrete_random_measure


def test_ghs017_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    W_i = rng.uniform(0, 1, 100)
    theta_i = rng.normal(0, 1, 100)
    result = ghosal_ch3_discrete_random_measure(W_i, theta_i)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "distribution" in result
    assert "total_mass" in result
    assert "method" in result
    # estimate is the weighted mean of theta_i with normalized weights
    w_sum = sum(W_i)
    expected_estimate = sum(wi * t for wi, t in zip(W_i, theta_i)) / w_sum
    assert abs(result["estimate"] - expected_estimate) < 1e-12
    # total_mass is sum of normalized weights = 1.0
    assert abs(result["total_mass"] - 1.0) < 1e-12


def test_ghs017_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    W_i = rng.uniform(0, 1, 100)
    theta_i = rng.normal(0, 1, 100)
    result = ghosal_ch3_discrete_random_measure(W_i, theta_i)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "distribution" in result
