"""Tests for morie.fn.permutations_count: recompute Morin (2016) from the formula."""

import math

from morie.fn.permutations_count import permutations_count


def test_factorial():
    for n in (0, 1, 6, 12):
        assert permutations_count(n)["permutations"] == math.factorial(n)
