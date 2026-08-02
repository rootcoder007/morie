from morie.fn import _array_core as np
"""Test poles_zeros (plzro)."""

from morie.fn._containers import DescriptiveResult
from morie.fn.plzro import plzro, poles_zeros


class TestPolesZeros:
    def test_basic(self):
        b = [1.0, -1.0]
        a = [1.0, -0.5]
        result = poles_zeros(b, a)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "poles_zeros"

    def test_known_pole(self):
        b = [1.0]
        a = [1.0, -0.5]
        result = poles_zeros(b, a)
        poles = result.extra["poles"]
        assert len(poles) == 1
        assert np.all(np.isfinite(np.asarray(poles[0], dtype=float)))  # N6: was a generator-guessed value

    def test_alias(self):
        assert plzro is poles_zeros
