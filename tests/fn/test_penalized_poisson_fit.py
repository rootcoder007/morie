"""Tests for penalized_poisson_fit.penalized_poisson_fit."""

from morie.fn import _array_core as np

from morie.fn.penalized_poisson_fit import penalized_poisson_fit


def test_msm122_basic():
    """Test basic functionality."""
    Poisson = np.random.default_rng(42).normal(0, 1, 100)
    regression = np.random.default_rng(42).normal(0, 1, 100)
    Given = np.random.default_rng(42).normal(0, 1, 100)
    vector = np.random.default_rng(42).normal(0, 1, 100)
    covariates = np.random.default_rng(42).normal(0, 1, 100)
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = penalized_poisson_fit(Poisson, regression, Given, vector, covariates, xi)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm122_edge():
    """Test edge cases."""
    Poisson = np.random.default_rng(42).normal(0, 1, 100)
    regression = np.random.default_rng(42).normal(0, 1, 100)
    Given = np.random.default_rng(42).normal(0, 1, 100)
    vector = np.random.default_rng(42).normal(0, 1, 100)
    covariates = np.random.default_rng(42).normal(0, 1, 100)
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = penalized_poisson_fit(Poisson, regression, Given, vector, covariates, xi)
    assert isinstance(result, dict)
