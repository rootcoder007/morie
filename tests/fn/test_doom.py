"""Tests for morie.fn.doom -- System failure cascade."""

from morie.fn._containers import DescriptiveResult
from morie.fn.doom import doom, failure_cascade


class TestDoom:
    def test_alias(self):
        assert doom is failure_cascade

    def test_series(self):
        result = failure_cascade([0.9, 0.9, 0.9], topology="series", seed=42)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - 0.729) < 0.05

    def test_parallel(self):
        result = failure_cascade([0.5, 0.5, 0.5], topology="parallel", seed=42)
        assert result.value > 0.8
