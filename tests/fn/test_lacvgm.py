"""Tests for morie.fn.lacvgm: expected values recomputed from the formulas."""

import math

from morie.fn.lacvgm import lacvgm

N = 9
_B = [[1.0 if abs(i - j) == 1 or {i, j} == {0, 8} or {i, j} == {2, 6} else 0.0 for j in range(N)] for i in range(N)]
W = [[v / sum(r) for v in r] for r in _B]
Y = [1.0, 2.4, 1.3, 3.1, 1.9, 2.2, 0.7, 2.8, 1.6]
X2 = [0.5, 0.9, 0.2, 1.4, 0.8, 1.1, 0.3, 1.2, 0.6]


def _lag(v, Wm=W):
    return [sum(a * b for a, b in zip(r, v)) for r in Wm]


def _consts(Wm):
    n = len(Wm)
    S0 = sum(map(sum, Wm))
    S1 = 0.5 * sum((Wm[i][j] + Wm[j][i]) ** 2 for i in range(n) for j in range(n))
    S2 = sum((sum(Wm[i]) + sum(Wm[j][i] for j in range(n))) ** 2 for i in range(n))
    return n, S0, S1, S2


def _upper(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


XY = [[0.0, 0.0], [1.0, 0.2], [2.1, 0.0], [0.1, 1.0], [1.2, 1.1], [2.0, 0.9], [0.3, 2.2], [1.1, 1.9], [2.3, 2.1]]


def test_matheron_bins():
    cut, nl = 2.0, 4
    w = cut / nl
    num, den, dist = [0.0] * nl, [0] * nl, [0.0] * nl
    for i in range(N):
        for j in range(i + 1, N):
            d = math.dist(XY[i], XY[j])
            if 0 < d <= cut:
                k = min(math.ceil(d / w) - 1, nl - 1)
                num[k] += (Y[i] - Y[j]) ** 2
                den[k] += 1
                dist[k] += d
    keep = [k for k in range(nl) if den[k]]
    r = lacvgm(Y, XY, n_lags=nl, cutoff=cut)
    assert r.extra["np"] == [den[k] for k in keep]
    assert max(abs(a - num[k] / (2 * den[k])) for a, k in zip(r.extra["gamma"], keep)) < 1e-13
    assert max(abs(a - dist[k] / den[k]) for a, k in zip(r.extra["dist"], keep)) < 1e-13


def test_default_cutoff_is_a_third_of_the_diagonal():
    r = lacvgm(Y, XY)
    assert abs(r.extra["cutoff"] - math.hypot(2.3, 2.2) / 3) < 1e-15
