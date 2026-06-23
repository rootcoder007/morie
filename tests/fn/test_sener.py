"""Test signal_energy (sener)."""

import numpy as np

from morie.fn._containers import DescriptiveResult
from morie.fn.sener import sener, signal_energy


class TestSignalEnergy:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0])
        result = signal_energy(x)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - 14.0) < 1e-10

    def test_zeros(self):
        x = np.zeros(5)
        assert signal_energy(x).value == 0.0

    def test_alias(self):
        assert sener is signal_energy
