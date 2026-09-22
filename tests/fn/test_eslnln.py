"""Tests for eslnln.esl_elastic_net."""

from morie.fn import _array_core as np

from morie.fn.eslnln import esl_elastic_net


def test_eslnln_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    lambda_ = 0.1
    alpha = 0.5
    result = esl_elastic_net(X, y, lambda_, alpha)
    assert isinstance(result, dict)
    for key in ("estimate", "beta", "n_nonzero", "active_set",
                "objective", "lambda", "alpha", "lambda1",
                "lambda2", "iterations", "converged", "n",
                "p", "method"):
        assert key in result
    assert result["lambda"] == 0.1
    assert result["alpha"] == 0.5
    # lambda1 = lambda * alpha, lambda2 = lambda * (1 - alpha)
    assert abs(result["lambda1"] - (0.1 * 0.5)) < 1e-12
    assert abs(result["lambda2"] - (0.1 * 0.5)) < 1e-12
    assert result["n"] == 100
    assert result["p"] == 5
    assert result["method"] == "elastic net, glmnet (lambda, alpha) parameterisation"


def test_eslnln_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    lambda_ = 0.0
    alpha = 0.05
    result = esl_elastic_net(X, y, lambda_, alpha)
    assert isinstance(result, dict)
    assert result["lambda"] == 0.0
    assert result["lambda1"] == 0.0
    assert result["lambda2"] == 0.0
    assert "estimate" in result
    assert "beta" in result
