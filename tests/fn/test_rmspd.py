"""Tests for morie.fn.rmspd -- RMSProp optimizer."""

from morie.fn import _array_core as np
from morie.fn._containers import DescriptiveResult
from morie.fn.rmspd import rmspd, rmsprop_optimize


class TestRmspd:
    def test_alias(self):
        assert rmspd is rmsprop_optimize

    def test_quadratic(self):
        def f(x):
            return np.sum(x**2)

        def g(x):
            return 2 * x

        r = rmsprop_optimize(f, g, np.array([5.0, 5.0]), lr=0.01, maxiter=3000)
        assert isinstance(r, DescriptiveResult)
        assert r.value < 1.0

    def test_converges(self):
        def f(x):
            return (x[0] - 2) ** 2 + (x[1] - 3) ** 2

        def g(x):
            return np.array([2 * (x[0] - 2), 2 * (x[1] - 3)])

        r = rmsprop_optimize(f, g, np.array([0.0, 0.0]), lr=0.01)
        assert r.value < 0.1
