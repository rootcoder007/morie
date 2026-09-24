"""Tests for msm055.mvsml_bayesian_regression_eq_6_5."""

import math

from morie.fn import _array_core as np

from morie.fn.msm055 import mvsml_bayesian_regression_eq_6_5


def test_msm055_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    Z = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    M = np.random.default_rng(44).normal(0.0, 1.0, (12, 3))
    G = np.array((M.T @ M).tolist()) / 12.0
    result = mvsml_bayesian_regression_eq_6_5(y, Z, G, n_iter=120, burn_in=20)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])


def test_msm055_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    Z = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    M = np.random.default_rng(44).normal(0.0, 1.0, (12, 3))
    G = np.array((M.T @ M).tolist()) / 12.0
    result = mvsml_bayesian_regression_eq_6_5(y, Z, G, n_iter=120, burn_in=20)
    assert isinstance(result, dict)
