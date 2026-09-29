"""Tests for morie.fn.sgcrh: the Cressie-Hawkins estimator recomputed bin by bin."""

import math

import pytest

from morie.fn.sgcrh import cressie_hawkins

XY = [[0.0, 0.0], [1.0, 0.2], [2.1, 0.0], [0.1, 1.0], [1.2, 1.1], [2.0, 0.9], [0.3, 2.2], [1.1, 1.9], [2.3, 2.1]]
Z = [1.0, 2.4, 1.3, 3.1, 1.9, 2.2, 0.7, 2.8, 1.6]


def _bins(cut, nl):
    w = cut / nl
    b = {}
    for i in range(9):
        for j in range(i + 1, 9):
            d = math.dist(XY[i], XY[j])
            if 0 < d <= cut:
                b.setdefault(min(math.ceil(d / w) - 1, nl - 1), []).append(abs(Z[i] - Z[j]) ** 0.5)
    return [b[k] for k in sorted(b)]


def test_cressie_1993_form():
    r = cressie_hawkins(Z, XY, lags=4, cutoff=2.0)
    for got, vals in zip(r.extra["gamma"], _bins(2.0, 4)):
        N = len(vals)
        want = (sum(vals) / N) ** 4 / (0.457 + 0.494 / N + 0.045 / N**2) / 2
        assert abs(got - want) < 1e-13


def test_gstat_form_drops_the_second_order_term():
    r = cressie_hawkins(Z, XY, lags=4, cutoff=2.0, method="gstat")
    for got, vals in zip(r.extra["gamma"], _bins(2.0, 4)):
        N = len(vals)
        assert abs(got - (sum(vals) / N) ** 4 / (0.457 + 0.494 / N) / 2) < 1e-13
    with pytest.raises(ValueError):
        cressie_hawkins(Z, XY, method="bad")
