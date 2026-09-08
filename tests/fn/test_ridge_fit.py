"""Tests for ridge_fit.ridge_fit."""

from morie.fn import _array_core as np

from morie.fn.ridge_fit import ridge_fit


def test_msm258_basic():
    """Test basic functionality."""
    colnames = np.random.default_rng(42).normal(0, 1, 100)
    results_i = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    Observed = np.random.default_rng(42).normal(0, 1, 100)
    Predicted = np.random.default_rng(42).normal(0, 1, 100)
    Trait = np.random.default_rng(42).normal(0, 1, 100)
    result = ridge_fit(colnames, results_i, c, Observed, Predicted, Trait)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm258_edge():
    """Test edge cases."""
    colnames = np.random.default_rng(42).normal(0, 1, 100)
    results_i = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    Observed = np.random.default_rng(42).normal(0, 1, 100)
    Predicted = np.random.default_rng(42).normal(0, 1, 100)
    Trait = np.random.default_rng(42).normal(0, 1, 100)
    result = ridge_fit(colnames, results_i, c, Observed, Predicted, Trait)
    assert isinstance(result, dict)
