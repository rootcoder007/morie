"""Tests for morie.fn.excess_score_factor: values recomputed from first principles."""

import math

from morie.fn.excess_score_factor import excess_score_factor


def test_identity():
    for r in (0.6, -0.3, 0.95):
        assert abs(excess_score_factor(r)["factor"] - (1 - r) / math.sqrt(1 - r * r)) < 1e-14
