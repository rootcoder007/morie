"""Tests for morie.fn.exponential_crossing_time: values recomputed from first principles."""

import math

from morie.fn.exponential_crossing_time import exponential_crossing_time


def test_crossing_condition():
    for rf, rs, q in ((0.2, 0.05, 4.0), (1.0, 0.1, 2.5)):
        t = exponential_crossing_time(rf, rs, q)["t"]
        assert abs(math.exp(-rf * t) - math.exp(-rs * t) / q) < 1e-15
