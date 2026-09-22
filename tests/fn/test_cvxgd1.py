"""Tests for cvxgd1.boyd_grad_proj."""

from morie.fn import _array_core as np

from morie.fn.cvxgd1 import boyd_grad_proj


def test_cvxgd1_basic():
    """Test basic functionality."""
    target = np.array([3.0, 4.0])
    f = lambda z: 0.5 * float(np.sum((np.asarray(z) - target) ** 2))
    grad_f = lambda z: np.asarray(z, dtype=float) - target
    x0 = np.array([0.0, 0.0])
    result = boyd_grad_proj(f, grad_f, x0, "ball", t=0.5, radius=1.0)
    assert isinstance(result, dict)
    # x should be on the unit ball, lying along the line from origin to target
    expected_x = target / float(np.linalg.norm(target))
    assert np.allclose(result["x"], expected_x, atol=1e-6)
    # Output keys documented in the docstring
    for key in ("x", "f", "n_iter", "converged", "feasible",
                "fixed_point_residual", "trajectory"):
        assert key in result
    # Every iterate lies in the ball (feasible throughout, not just at end)
    assert bool(np.all(np.linalg.norm(result["trajectory"], axis=1) <= 1.0 + 1e-9))
    # Fixed-point residual is the first-order optimality certificate
    assert result["converged"] is True
    assert result["fixed_point_residual"] < 1e-6


def test_cvxgd1_edge():
    """Test edge cases."""
    # Simplex projection: iterates must remain a probability vector.
    coeffs = np.array([1.0, 2.0, 3.0])
    f = lambda z: float(np.asarray(z) @ coeffs)
    grad_f = lambda z: coeffs.copy()
    x0 = np.array([0.4, 0.4, 0.2])
    result = boyd_grad_proj(f, grad_f, x0, "simplex", t=0.1)
    assert isinstance(result, dict)
    assert bool(abs(float(np.sum(result["x"])) - 1.0) < 1e-9)
    assert bool(np.all(np.asarray(result["x"]) >= -1e-12))
