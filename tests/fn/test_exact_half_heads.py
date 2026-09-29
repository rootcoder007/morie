"""Tests for morie.fn.exact_half_heads: values recomputed from first principles."""

import math

from morie.fn.exact_half_heads import exact_half_heads


def test_binomial_count():
    for n in (1, 5, 20):
        want = (
            sum(1 for k in range(2 ** (2 * n)) if bin(k).count("1") == n) / 4**n
            if n <= 5
            else math.comb(2 * n, n) / 4**n
        )
        assert abs(exact_half_heads(n)["probability"] - want) < 1e-15
