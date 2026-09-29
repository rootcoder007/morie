"""Tests for morie.fn.hockey_stick: recompute Morin (2016) from the formula."""

import math

from morie.fn.hockey_stick import hockey_stick


def test_identity():
    for n, k in ((7, 3), (10, 1), (6, 6), (12, 5)):
        r = hockey_stick(n, k)
        assert r["sum"] == sum(math.comb(j, k - 1) for j in range(k - 1, n))
        assert r["binomial"] == math.comb(n, k)
        assert r["identity_holds"]
