"""Tests for cvxprg.boyd_proximal_grad."""

from morie.fn import _array_core as np

from morie.fn.cvxprg import boyd_proximal_grad


def test_cvxprg_basic():
    """Test basic functionality with a simple smooth quadratic."""
    rng = np.random.default_rng(42)

    # Smooth part: f(z) = 0.5 * ||z - b||^2, grad_f(z) = z - b
    b = rng.normal(0, 1, 10)
    f = lambda z: 0.5 * float(np.sum((np.asarray(z) - b) ** 2))
    grad_f = lambda z: np.asarray(z) - b

    # Nonsmooth part: h(z) = lam * ||z||_1, prox = soft thresholding
    lam = 0.5
    h = lambda z: lam * float(np.sum(np.abs(np.asarray(z))))
    soft = lambda v, s: np.sign(v) * np.maximum(np.abs(v) - s * lam, 0.0)

    x0 = rng.normal(0, 1, 10)
    t = 1.0

    result = boyd_proximal_grad(f, grad_f, soft, x0, t=t, h=h)

    # Documented return keys
    assert "x" in result
    assert "f" in result
    assert "objective" in result
    assert "n_iter" in result
    assert "converged" in result
    assert "n_zero" in result
    assert "objective_path" in result

    # Shapes: x matches x0
    assert np.shape(result["x"]) == np.shape(x0)

    # n_zero is an integer count of exactly-zero entries
    assert isinstance(result["n_zero"], int)
    assert 0 <= result["n_zero"] <= int(np.size(x0))
    assert int(np.sum(np.abs(np.asarray(result["x"])) <= 1e-12)) == result["n_zero"]

    # f(x) and objective agree: objective = f(x) + h(x)
    x = np.asarray(result["x"])
    expected_objective = float(0.5 * float(np.sum((x - b) ** 2))) + float(h(x))
    assert result["objective"] == expected_objective
    assert result["f"] == float(0.5 * float(np.sum((x - b) ** 2)))

    # The solver started where we told it to
    x_path = result["objective_path"]
    assert np.shape(x_path)[0] >= 2

    # All entries are finite
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(np.asarray(x_path)))


def test_cvxprg_edge():
    """Test that providing x0 at the optimum keeps the objective flat."""
    rng = np.random.default_rng(7)

    b = rng.normal(0, 1, 5)
    f = lambda z: 0.5 * float(np.sum((np.asarray(z) - b) ** 2))
    grad_f = lambda z: np.asarray(z) - b

    lam = 0.1
    h = lambda z: lam * float(np.sum(np.abs(np.asarray(z))))
    soft = lambda v, s: np.sign(v) * np.maximum(np.abs(v) - s * lam, 0.0)

    # Start at the unconstrained minimizer of f; h adds an L1 penalty.
    # The first iterate is the prox-soft-thresholded version of b.
    x0 = b.copy()
    result = boyd_proximal_grad(f, grad_f, soft, x0, t=0.5, h=h)

    assert isinstance(result, dict)
    assert "objective" in result
    assert "x" in result

    # objective_path must have at least two entries (initial + first iter)
    path = np.asarray(result["objective_path"])
    assert np.shape(path)[0] >= 2

    # n_iter and converged are sensible
    assert isinstance(result["n_iter"], int)
    assert result["n_iter"] >= 1
    assert isinstance(result["converged"], bool)
