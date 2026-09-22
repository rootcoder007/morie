"""Tests for cvxnda.boyd_newton_decrement."""

from morie.fn import _array_core as np

from morie.fn.cvxnda import boyd_newton_decrement


def test_cvxnda_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 4
    grad = rng.normal(0, 1, n)
    # Construct a symmetric positive-definite Hessian
    M = rng.normal(0, 1, (n, n))
    hess = M @ M.T + n * np.eye(n)

    result = boyd_newton_decrement(grad, hess)
    assert isinstance(result, dict)
    assert "decrement" in result
    assert "suboptimality" in result
    assert "grad_norm" in result

    # Independent computation of lambda = sqrt(g^T H^-1 g)
    g = np.asarray(grad, dtype=float).ravel()
    Hm = np.asarray(hess, dtype=float)
    quad = float(g @ np.linalg.solve(Hm, g))
    expected_lam = float(np.sqrt(quad))
    expected_subopt = expected_lam ** 2 / 2
    expected_gnorm = float(np.linalg.norm(g))

    assert abs(result["decrement"] - expected_lam) < 1e-10
    assert abs(result["suboptimality"] - expected_subopt) < 1e-10
    assert abs(result["grad_norm"] - expected_gnorm) < 1e-10
    assert abs(result["suboptimality"] - expected_lam ** 2 / 2) < 1e-10


def test_cvxnda_edge():
    """Test edge cases."""
    # Quadratic case f(x) = 0.5 x'Ax with x = (5, -2), A = [[4, 1], [1, 3]]
    # The suboptimality lambda^2/2 must equal f(x) exactly.
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    x = np.array([5.0, -2.0])
    grad = A @ x
    result = boyd_newton_decrement(grad, A)

    assert isinstance(result, dict)
    assert "decrement" in result
    assert "suboptimality" in result
    assert "grad_norm" in result

    f = 0.5 * float(x @ A @ x)
    assert abs(result["suboptimality"] - f) < 1e-10

    # Affine invariance: rescale x -> S x with diagonal S.
    S = np.diag([10.0, 0.1])
    result2 = boyd_newton_decrement(S @ grad, S @ A @ S)
    assert abs(result2["decrement"] - result["decrement"]) < 1e-9

    # Zero gradient yields a zero decrement and suboptimality.
    zero_grad = np.zeros(2)
    result3 = boyd_newton_decrement(zero_grad, A)
    assert abs(result3["decrement"]) < 1e-12
    assert abs(result3["suboptimality"]) < 1e-12
    assert abs(result3["grad_norm"]) < 1e-12
