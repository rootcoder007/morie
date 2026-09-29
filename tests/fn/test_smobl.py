"""Test mobility (smobl)."""

from morie.fn import _array_core as np
from morie.fn._containers import DescriptiveResult
from morie.fn.smobl import mobility, smobl


class TestMobility:
    def test_positive(self):
        x = np.sin(np.linspace(0, 4 * np.pi, 100))
        result = mobility(x)
        assert isinstance(result, DescriptiveResult)
        assert result.value > 0

    def test_constant(self):
        x = np.ones(10)
        assert mobility(x).value == 0.0

    def test_alias(self):
        assert smobl is mobility


def test_mobility_recomputed():
    import math

    import pytest

    x = [0.5, 1.2, -0.3, 0.8, 2.0, -1.1]

    def v0(a):
        m = sum(a) / len(a)
        return sum((t - m) ** 2 for t in a) / len(a)

    d = [x[i + 1] - x[i] for i in range(5)]
    assert mobility(x).value == pytest.approx(math.sqrt(v0(d) / v0(x)), rel=1e-13)
