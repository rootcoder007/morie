"""Test signal_power (spowr)."""

from morie.fn import _array_core as np

from morie.fn._containers import DescriptiveResult
from morie.fn.spowr import signal_power, spowr


class TestSignalPower:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0])
        result = signal_power(x)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - 14.0 / 3.0) < 1e-10

    def test_alias(self):
        assert spowr is signal_power
