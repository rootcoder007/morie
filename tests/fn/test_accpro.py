"""Tests for morie.fn.accpro -- Acceleration profile."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn._containers import DescriptiveResult
from morie.fn.accpro import acceleration_profile, accpro


class TestAccpro:
    def test_alias(self):
        assert accpro is acceleration_profile

    def test_constant_velocity(self):
        df = pd.DataFrame({"x": np.ones(20) * 5.0})
        result = acceleration_profile(df, dt=1.0)
        assert isinstance(result, DescriptiveResult)
        assert result.value < 0.01

    def test_accelerating(self):
        df = pd.DataFrame({"x": np.arange(10, dtype=float) ** 2})
        result = acceleration_profile(df, dt=1.0)
        assert result.value > 0
