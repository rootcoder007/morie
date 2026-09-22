"""Tests for eslrdg.esl_ridge."""

from morie.fn import _array_core as np

from morie.fn.eslrdg import esl_ridge


def test_eslrdg_basic():
    """Test basic functionality."""
    rng_X = np.random.default_rng(42)
    X = rng_X.normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    lambda_ = 0.5
    result = esl_ridge(X, y, lambda_)
    assert isinstance(result, dict)
    assert "beta" in result
    assert "estimate" in result

    # Independent computation of the closed-form ridge solution.
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    p = X_arr.shape[1]
    I = np.eye(p)
    # Build the penalty matrix: column 0 (intercept) is left unpenalised.
    P = I.copy()
    P[0, 0] = 0.0
    A = X_arr.T @ X_arr + lambda_ * P
    beta_expected = np.linalg.solve(A, X_arr.T @ y_arr)

    beta = np.asarray(result["beta"])
    assert beta.shape == (p,)
    assert np.allclose(beta, beta_expected, atol=1e-8)

    # estimate is documented to be the first coefficient.
    assert np.isclose(result["estimate"], float(beta_expected[0]), atol=1e-8)


def test_eslrdg_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    lambda_ = 0.0
    result = esl_ridge(X, y, lambda_)
    assert isinstance(result, dict)
    assert "beta" in result
    assert np.asarray(result["beta"]).shape == (5,)
