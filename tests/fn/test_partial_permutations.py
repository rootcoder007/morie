"""Tests for morie.fn.partial_permutations: recompute Morin (2016) from the formula."""

import math

from morie.fn.partial_permutations import partial_permutations


def test_falling_factorial():
    for N, n in ((8, 3), (5, 5), (9, 0)):
        assert partial_permutations(N, n)["partial_permutations"] == math.factorial(N) // math.factorial(N - n)
    assert partial_permutations(8, 3)["partial_permutations"] == 8 * 7 * 6
