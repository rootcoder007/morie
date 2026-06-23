"""Test zero_crossing_rate."""

import numpy as np

from morie.fn._containers import DescriptiveResult
from morie.fn.zcr import alias, zero_crossing_rate


class TestZeroCrossingRate:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = zero_crossing_rate(x)
        assert isinstance(result, DescriptiveResult)

    def test_value_range(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = zero_crossing_rate(x)
        assert isinstance(result.value, float)
        assert result.value >= 0.0

    def test_name(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = zero_crossing_rate(x)
        assert result.name == "zero_crossing_rate"

    def test_constant_signal_zero_zcr(self):
        x = np.ones(256)
        result = zero_crossing_rate(x)
        assert result.value == 0.0

    def test_alias(self):
        assert alias is zero_crossing_rate
