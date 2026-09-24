"""Tests for eslemg.esl_em_gmm."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.eslemg import esl_em_gmm


def test_eslemg_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5))
    k = 5
    result = esl_em_gmm(X, k)
    assert isinstance(result, dict)
    # All keys promised by the return statement
    for key in ("pi", "mu", "sigma", "resp", "labels", "loglik",
                "loglik_path", "n_iter", "converged", "aic", "bic"):
        assert key in result
    # Shape checks
    mu = result["mu"]
    assert len(mu) == k
    for row in mu:
        assert len(row) == 5
    sigma = result["sigma"]
    assert len(sigma) == k
    for mat in sigma:
        assert len(mat) == 5
        for row in mat:
            assert len(row) == 5
    resp = result["resp"]
    assert len(resp) == 100
    for row in resp:
        assert len(row) == k
    assert len(result["labels"]) == 100
    assert len(result["pi"]) == k
    assert math.isfinite(result["loglik"])
    # Log-likelihood must be non-decreasing across EM iterations
    path = result["loglik_path"]
    diffs = [path[i + 1] - path[i] for i in range(len(path) - 1)]
    assert all(d > -1e-8 for d in diffs)


def test_eslemg_edge():
    """Test edge cases."""
    # k greater than the number of observations is documented as invalid
    X = [[1.0], [2.0]]
    with pytest.raises(ValueError):
        esl_em_gmm(X, k=5)
