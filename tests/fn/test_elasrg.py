"""Tests for elasrg.elastic_net_regression."""

from morie.fn import _array_core as np

from morie.fn.elasrg import elastic_net_regression


def test_elasrg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    lambda1 = 0.1
    lambda2 = 0.2
    result = elastic_net_regression(y, X, lambda1, lambda2)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result

    # Verify the documented return-key names are present.
    expected_keys = {
        "estimate", "beta", "n_nonzero", "objective",
        "lambda1", "lambda2", "equivalent_lambda",
        "equivalent_alpha", "iterations", "converged",
        "n", "p", "method",
    }
    for key in expected_keys:
        assert key in result, f"missing key: {key}"

    # Shape / dtype sanity checks on documented fields.
    yv = np.asarray(y, dtype=float).ravel()
    n = yv.shape[0]
    assert result["n"] == n
    assert result["p"] == X.shape[1]
    assert result["lambda1"] == float(lambda1)
    assert result["lambda2"] == float(lambda2)

    # equivalent_lambda and equivalent_alpha from the documented
    # mapping: total = lambda1 + lambda2; alpha = lambda1 / total.
    lam1 = float(lambda1)
    lam2 = float(lambda2)
    total = lam1 + lam2
    assert result["equivalent_lambda"] == total
    if total > 0:
        assert abs(result["equivalent_alpha"] - lam1 / total) < 1e-12
    else:
        import math
        assert math.isnan(result["equivalent_alpha"])

    # Convergence is documented as a boolean.
    assert isinstance(result["converged"], bool)
    assert isinstance(result["iterations"], int)

    # estimate and beta must agree on the first coefficient.
    assert result["estimate"] == result["beta"][0]

    # n_nonzero must equal the count of non-zero entries in beta.
    assert result["n_nonzero"] == int(sum(1 for b in result["beta"] if b != 0))


def test_elasrg_edge():
    """Test edge cases: both penalties zero (pure least squares)."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    lambda1 = 0.0
    lambda2 = 0.0
    result = elastic_net_regression(y, X, lambda1, lambda2)
    assert isinstance(result, dict)

    # With no penalisation the coefficient should match the
    # ordinary-least-squares solution on the same (X, y).
    # Solve normal equations X^T X beta = X^T y via plain arithmetic:
    # beta_ols = (X^T X)^{-1} (X^T y), using np.linalg.solve on the
    # Gram matrix so we don't depend on any function under test.
    Xm = np.asarray(X, dtype=float)
    yv = np.asarray(y, dtype=float).ravel()
    XtX = Xm.T @ Xm
    Xty = Xm.T @ yv
    beta_ols = np.linalg.solve(XtX, Xty)

    for j in range(X.shape[1]):
        assert abs(result["beta"][j] - float(beta_ols[j])) < 1e-6

    # equivalent_alpha is undefined (0/0) when both penalties are 0.
    import math
    assert math.isnan(result["equivalent_alpha"])
    assert result["equivalent_lambda"] == 0.0
