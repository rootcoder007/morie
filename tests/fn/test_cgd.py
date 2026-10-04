"""Tests for morie.fn.cgd — conjugate gradient descent."""

from morie.fn import _array_core as np
from morie.fn.cgd import conjugate_gradient


class TestConjugateGradient:
    def test_returns_result(self):
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        def grad_f(x):
            return np.array([2 * x[0], 2 * x[1]])

        x0 = np.array([5.0, -3.0])
        res = conjugate_gradient(f, grad_f, x0)
        assert "x_opt" in res.extra
        assert "f_opt" in res.extra
        assert isinstance(res.extra["f_opt"], float)

    def test_f_opt_leq_initial(self):
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        def grad_f(x):
            return np.array([2 * x[0], 2 * x[1]])

        x0 = np.array([1.0, 1.0])
        res = conjugate_gradient(f, grad_f, x0)
        assert res.extra["f_opt"] <= f(x0) + 1e-10

    def test_n_iter_positive(self):
        def f(x):
            return (x[0] - 1) ** 2 + (x[1] + 2) ** 2

        def grad_f(x):
            return np.array([2 * (x[0] - 1), 2 * (x[1] + 2)])

        x0 = np.array([0.0, 0.0])
        res = conjugate_gradient(f, grad_f, x0)
        assert res.extra["n_iter"] > 0
        assert isinstance(res.extra["final_grad_norm"], float)
