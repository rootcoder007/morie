"""Tests for eslmix.esl_gaussian_mixture."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.eslmix import esl_gaussian_mixture


def test_eslmix_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    result = esl_gaussian_mixture(X, k=2)
    assert isinstance(result, dict)
    # Check that all documented keys are present
    expected_keys = {
        "density", "log_density", "pi", "mu", "sigma",
        "loglik", "aic", "bic", "resp", "labels", "method"
    }
    assert expected_keys.issubset(result.keys())
    # Shapes
    n = 40
    p = 3
    k = 2
    assert len(result["density"]) == n
    assert len(result["log_density"]) == n
    assert len(result["pi"]) == k
    assert len(result["mu"]) == k
    for m in result["mu"]:
        assert len(m) == p
    assert len(result["sigma"]) == k
    for s in result["sigma"]:
        assert len(s) == p
        for row in s:
            assert len(row) == p
    assert len(result["resp"]) == n
    assert len(result["resp"][0]) == k
    assert len(result["labels"]) == n
    # Finite scalars
    assert math.isfinite(result["loglik"])
    assert math.isfinite(result["aic"])
    assert math.isfinite(result["bic"])
    # Density should be positive and finite
    assert all(math.isfinite(d) and d >= 0 for d in result["density"])
    assert all(math.isfinite(ld) for ld in result["log_density"])


def test_eslmix_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    # Correct newdata shape
    newdata = rng.normal(0, 1, (20, 3))
    result = esl_gaussian_mixture(X, k=2, newdata=newdata)
    assert isinstance(result, dict)
    assert len(result["density"]) == 20
    # Mismatched columns should raise
    with pytest.raises(ValueError):
        esl_gaussian_mixture(X, k=2, newdata=rng.normal(0, 1, (20, 4)))
