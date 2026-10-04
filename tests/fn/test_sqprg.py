"""Tests for morie.fn.sqprg -- SQP optimizer."""

from morie.fn import _array_core as np
from morie.fn._containers import DescriptiveResult
from morie.fn.sqprg import sqp_optimize, sqprg


class TestSqprg:
    def test_alias(self):
        assert sqprg is sqp_optimize

    def test_equality_constrained(self):
        def f(x):
            return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2

        def g(x):
            return np.array([2 * (x[0] - 1), 2 * (x[1] - 2.5)])

        cons = [lambda x: x[0] + x[1] - 3]
        r = sqp_optimize(f, g, cons, np.array([0.0, 0.0]))
        assert isinstance(r, DescriptiveResult)
        assert abs(r.extra["x"][0] + r.extra["x"][1] - 3.0) < 0.1
