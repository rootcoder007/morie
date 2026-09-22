"""Tests for blue_blup_via_v.blue_blup_via_v."""

from morie.fn import _array_core as np

from morie.fn.blue_blup_via_v import blue_blup_via_v


def test_msm240_basic():
    """Test basic functionality.

    The model is: y = X beta + Z u + eps, with u ~ N(0, Sigma),
    eps ~ N(0, R). V = Z Sigma Z' + R. The BLUE is
        beta = (X' V^-1 X)^-1 X' V^-1 y
    and the BLUP is
        u = Sigma Z' V^-1 (y - X beta).

    The function exposes `beta`, `u`, `estimate` (first element of beta)
    and a method label.
    """
    rng = np.random.default_rng(42)
    n = 50
    p = 2  # number of fixed effects (columns of X)
    q = 3  # number of random effects (columns of Z = rows of Sigma)

    # Design matrix for fixed effects
    X = rng.normal(0, 1, (n, p))
    # Design matrix for random effects
    Z = rng.normal(0, 1, (n, q))
    # Response vector
    y = rng.normal(0, 1, n)
    # Covariance of random effects u (q, q), positive (semi-)definite
    A = rng.normal(0, 1, (q, q))
    Sigma = A @ A.T + np.eye(q)
    # Residual covariance R (n, n), positive (semi-)definite; diagonal here
    R = np.eye(n)

    result = blue_blup_via_v(X, Z, y, Sigma, R)

    # The result must be a mapping/dict-like object with the documented keys
    assert hasattr(result, "__getitem__")
    assert "estimate" in result
    assert "blue" in result
    assert "blup" in result
    assert "method" in result

    # Shapes of BLUE and BLUP follow from the documented formula
    beta = np.asarray(result["blue"])
    u = np.asarray(result["blup"])
    assert beta.shape == (p,)
    assert u.shape == (q,)

    # Independent recomputation of beta and u from the documented formula
    V = Z @ Sigma @ Z.T + R
    # V is square (n, n); invert it explicitly
    V_inv = np.linalg.inv(V)
    beta_hat = np.linalg.inv(X.T @ V_inv @ X) @ (X.T @ V_inv @ y)
    u_hat = Sigma @ Z.T @ V_inv @ (y - X @ beta_hat)

    # Compare to the function's outputs element-wise
    assert np.allclose(beta, beta_hat)
    assert np.allclose(u, u_hat)

    # The "estimate" key exposes the first fixed effect
    assert np.allclose(np.asarray(result["estimate"]), beta_hat[0])


def test_msm240_edge():
    """Test edge case: only four positional arguments (R defaults to None).

    With R omitted, the implementation is expected to use the default
    residual covariance. We still must satisfy the documented shape
    constraints (Sigma must match the number of random effects implied
    by Z, i.e. Sigma is q x q where Z is n x q).
    """
    rng = np.random.default_rng(123)
    n = 20
    p = 1
    q = 2

    X = rng.normal(0, 1, (n, p))
    Z = rng.normal(0, 1, (n, q))
    y = rng.normal(0, 1, n)
    A = rng.normal(0, 1, (q, q))
    Sigma = A @ A.T + np.eye(q)

    # No R supplied: rely on the function's default behaviour
    result = blue_blup_via_v(X, Z, y, Sigma)

    assert hasattr(result, "__getitem__")
    assert "estimate" in result
    assert "blue" in result
    assert "blup" in result
    assert "method" in result

    beta = np.asarray(result["blue"])
    u = np.asarray(result["blup"])
    assert beta.shape == (p,)
    assert u.shape == (q,)
