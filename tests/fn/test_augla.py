"""Tests for morie.fn.augla -- Augmented Lagrangian."""

from morie.fn import _array_core as np
from morie.fn._containers import DescriptiveResult
from morie.fn.augla import augla, augmented_lagrangian


class TestAugla:
    def test_alias(self):
        assert augla is augmented_lagrangian

    def test_constrained_min(self):
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        def g(x):
            return np.array([2 * x[0], 2 * x[1]])

        cons = [lambda x: x[0] + x[1] - 1]
        r = augmented_lagrangian(f, g, cons, np.array([2.0, 2.0]))
        assert isinstance(r, DescriptiveResult)
        assert abs(r.extra["x"][0] + r.extra["x"][1] - 1.0) < 0.1
        assert abs(r.extra["x"][0] - 0.5) < 0.2
