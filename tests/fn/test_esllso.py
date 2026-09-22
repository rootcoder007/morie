"""Tests for esllso.esl_lasso."""

from morie.fn import _array_core as np

from morie.fn.esllso import esl_lasso


def test_esllso_basic():
    """Test basic functionality with a scalar penalty and random design."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    X = rng_x.normal(0, 1, (100, 5))
    y = rng_y.normal(0, 1, 100)
    lambda_ = 0.1  # documented as a non-negative float scalar
    result = esl_lasso(X, y, lambda_)
    assert isinstance(result, dict)
    # Documented return keys
    for key in ("estimate", "beta", "n_nonzero", "active_set",
                "objective", "iterations", "converged",
                "lambda", "n", "p", "method"):
        assert key in result, f"missing documented key: {key}"
    assert result["n"] == 100
    assert result["p"] == 5
    assert result["lambda"] == lambda_
    assert len(result["beta"]) == 5
    assert result["n_nonzero"] == len(result["active_set"])
    # n_nonzero must equal the count of non-zero entries in beta
    assert result["n_nonzero"] == sum(1 for b in result["beta"] if b != 0.0)
    # estimate is the first coefficient
    assert result["estimate"] == result["beta"][0]


def test_esllso_edge():
    """Test the orthonormal design example from the docstring."""
    # Orthonormal columns => soft-thresholded OLS with no penalisation of
    # the constant column (the intercept in the docstring example is a
    # constant column because both entries of column 0 are 1.0? No: column 0
    # is [1, 0] which is not constant, but the docstring example uses it as
    # an "intercept-like" coefficient). For column j with ||x_j||^2 = 1 and
    # no penalisation, beta_j = X_j^T y (OLS); with penalisation, beta_j =
    # soft(rho, lambda) where rho = X_j^T y.
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([3.0, -1.0])

    lam = 0.5
    result = esl_lasso(X, y, lam)
    assert isinstance(result, dict)
    assert result["n"] == 2
    assert result["p"] == 2
    assert result["lambda"] == lam
    # Neither column is constant (ptp = 1), so both are penalised.
    # ||x_j||^2 = 1, rho_j = X_j^T y, beta_j = sign(rho) * max(|rho|-lam, 0).
    expected = [
        (1.0 if 3.0 >= 0 else -1.0) * max(abs(3.0) - lam, 0.0) / 1.0,
        (1.0 if -1.0 >= 0 else -1.0) * max(abs(-1.0) - lam, 0.0) / 1.0,
    ]
    assert result["beta"] == [round(e, 12) for e in expected]

    # Larger penalty should zero out the second coefficient (|rho|=1 < lam=1.5)
    lam2 = 1.5
    result2 = esl_lasso(X, y, lam2)
    assert result2["active_set"] == [0]
    assert result2["beta"][1] == 0.0
    # And the active coefficient is soft-thresholded: sign(3)*(|3|-1.5)/1 = 1.5
    assert result2["beta"][0] == 1.5
    assert result2["n_nonzero"] == 1
