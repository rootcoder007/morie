"""Tests for morie.fn.adamm -- Adam optimizer."""

from morie.fn import _array_core as np
from morie.fn._containers import DescriptiveResult
from morie.fn.adamm import adam_optimize, adamm


class TestAdamm:
    def test_alias(self):
        assert adamm is adam_optimize

    def test_rosenbrock_2d(self):
        def f(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        def g(x):
            return np.array(
                [
                    -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
                    200 * (x[1] - x[0] ** 2),
                ]
            )

        r = adam_optimize(f, g, np.array([0.0, 0.0]), lr=0.01, maxiter=5000)
        assert isinstance(r, DescriptiveResult)
        assert r.value < 1.0

    def test_quadratic(self):
        def f(x):
            return np.sum(x**2)

        def g(x):
            return 2 * x

        r = adam_optimize(f, g, np.array([5.0, 5.0]), lr=0.01, maxiter=5000)
        assert r.value < 50.0
        assert r.value < f(np.array([5.0, 5.0]))
