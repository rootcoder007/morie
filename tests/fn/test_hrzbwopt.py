"""Tests for hrzbwopt.horowitz_optimal_bandwidth_kde."""

from morie.fn import _array_core as np

from morie.fn.hrzbwopt import horowitz_optimal_bandwidth_kde


def test_hrzbwopt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_optimal_bandwidth_kde(x)
    assert isinstance(result, dict)
    assert "h_opt" in result


def test_hrzbwopt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_optimal_bandwidth_kde(x)
    assert isinstance(result, dict)
