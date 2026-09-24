"""Tests for eslvbt.esl_var_beta_hat."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.eslvbt import esl_var_beta_hat


def test_eslvbt_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    sigma2 = 2.5
    result = esl_var_beta_hat(X, sigma2)
    assert isinstance(result, dict)
    for key in ("estimate", "covariance", "variances", "se", "n", "p", "method"):
        assert key in result
    assert result["n"] == 40
    assert result["p"] == 3
    assert len(result["variances"]) == 3
    assert len(result["se"]) == 3
    assert len(result["covariance"]) == 9
    assert math.isfinite(result["estimate"])
    for v in result["variances"]:
        assert math.isfinite(v)
        assert v > 0
    for s in result["se"]:
        assert math.isfinite(s)
        assert s > 0
    for c in result["covariance"]:
        assert math.isfinite(c)
    assert isinstance(result["method"], str)


def test_eslvbt_edge():
    """Test edge cases: invalid sigma2."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    with pytest.raises(ValueError):
        esl_var_beta_hat(X, 0.0)
    with pytest.raises(ValueError):
        esl_var_beta_hat(X, -1.0)
