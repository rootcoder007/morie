"""Tests for morie.fn.sd_fair_coin_avg: values recomputed from first principles."""

import math

from morie.fn.sd_fair_coin_avg import sd_fair_coin_avg


def test_sd_of_the_heads_fraction():
    n = 12
    # exact binomial variance of k / n
    var = sum(math.comb(n, k) / 2**n * (k / n - 0.5) ** 2 for k in range(n + 1))
    assert abs(sd_fair_coin_avg(n)["sd_avg"] - math.sqrt(var)) < 1e-15
