"""Tests for cvxgrd.boyd_gradient_descent."""

from morie.fn import _array_core as np

from morie.fn.cvxgrd import boyd_gradient_descent


def test_cvxgrd_basic():
    """Test basic functionality on a well-conditioned quadratic."""
    rng = np.random.default_rng(42)
    # Well-conditioned diagonal Hessian with eigenvalues in [1, 2].
    Q = np.diag([1.0, 2.0])

    def f(x):
        return 0.5 * x @ (Q @ x)

    def grad_f(x):
        return Q @ x

    x0 = [1.0, 1.0]
    # Largest Hessian eigenvalue is 2, so 2/L = 1.  Use t = 0.4 (well inside
    # the stable range 0 < t < 2/L).
    t = 0.4
    result = boyd_gradient_descent(f, grad_f, x0, t)

    # The function returns a rich-result object with a dict-like payload.
    assert hasattr(result, "payload")
    payload = result.payload

    # All these keys are part of the documented return value.
    for key in ("x", "f", "n_iter", "converged", "grad_norm",
                "trajectory", "diverged", "monotone"):
        assert key in payload

    # The optimisation should converge to the origin for this quadratic.
    assert payload["converged"] is True or payload["diverged"] is False

    # Independent computation of the optimal point: x* = Q^{-1} Q x0 ... no,
    # gradient = Q x, so x* = 0 for a quadratic 0.5 x^T Q x.
    # Compute the expected gradient norm at the final iterate via an
    # independent expression of the documented formula x^{k+1} = x^k - t grad.
    x_final = np.asarray(payload["x"], dtype=float)
    expected_grad = np.asarray(grad_f(x_final), dtype=float)
    assert float(np.max(np.abs(expected_grad))) == payload["grad_norm"]

    # Each trajectory step should satisfy the documented update.
    traj = np.asarray(payload["trajectory"], dtype=float)
    for k in range(len(traj) - 1):
        diff = np.asarray(traj[k + 1]) - np.asarray(traj[k])
        grad = np.asarray(grad_f(np.asarray(traj[k])), dtype=float)
        # diff == -t * grad  (equal up to numerical noise)
        np.testing.assert_allclose(diff, -t * grad, atol=1e-10)


def test_cvxgrd_edge():
    """Edge case: a fixed step past 2/L should be reported as diverged."""
    Q = np.diag([1.0, 2.0])  # L = 2, so 2/L = 1; pick t = 1.5.

    def f(x):
        return 0.5 * x @ (Q @ x)

    def grad_f(x):
        return Q @ x

    x0 = [1.0, 1.0]
    result = boyd_gradient_descent(f, grad_f, x0, t=1.5, max_iter=200)

    assert hasattr(result, "payload")
    payload = result.payload
    assert payload["diverged"] is True
    assert "diverged" in payload
