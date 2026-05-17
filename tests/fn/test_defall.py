"""Nature does not hurry, yet everything is accomplished. — Lao Tzu"""

import numpy as np
from morie.fn.defall import defense_allocation, defall
from morie.fn._containers import DescriptiveResult


class TestDefall:
    def test_alias(self):
        assert defall is defense_allocation

    def test_proportional(self):
        result = defense_allocation([10, 20, 30], resources=100)
        assert isinstance(result, DescriptiveResult)
        alloc = result.value
        assert abs(sum(alloc) - 100) < 0.01

    def test_equal_threats(self):
        result = defense_allocation([5, 5, 5, 5], resources=100)
        for a in result.value:
            assert abs(a - 25.0) < 0.01
