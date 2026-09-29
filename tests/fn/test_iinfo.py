"""Tests for iinfo.item_information: the 3PL information recomputed."""

import math

from morie.fn.iinfo import item_information


def test_three_pl_information():
    th, a, b, c = [-1.0, 0.0, 1.5], 1.4, 0.3, 0.2
    r = item_information(th, a=a, b=b, c=c)
    for k, t in enumerate(th):
        ps = 1 / (1 + math.exp(-a * (t - b)))
        p = c + (1 - c) * ps
        ref = (a * (1 - c) * ps * (1 - ps)) ** 2 / (p * (1 - p))
        assert abs(r["info"][k] - ref) < 1e-15


def test_two_pl_is_a2pq():
    r = item_information([0.7], a=2.0, b=0.1)
    p = 1 / (1 + math.exp(-2.0 * 0.6))
    assert abs(r["info"][0] - 4 * p * (1 - p)) < 1e-15
