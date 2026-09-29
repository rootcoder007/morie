"""Tests for morie.fn.sarsc: every expected value is recomputed from the formula."""

import math

from morie.fn.sarsc import sarsc

N = 8
_B = [[1.0 if abs(i - j) == 1 or {i, j} == {0, 7} or {i, j} == {2, 5} else 0.0 for j in range(N)] for i in range(N)]
W = [[v / sum(r) for v in r] for r in _B]
X = [[1.0, ((i * 7) % 11) / 5] for i in range(N)]
Y = [1 + 2 * X[i][1] + ((i * 3) % 5 - 2) / 4 + 0.3 * sum(W[i][j] * X[j][1] for j in range(N)) for i in range(N)]
E = [((i * 4) % 7 - 3) / 5 for i in range(N)]
W3 = [[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]]


def _resid(v):
    # residual of v on [1, x] by the closed-form simple regression
    x = [row[1] for row in X]
    mx, mv = sum(x) / N, sum(v) / N
    b1 = sum((a - mx) * (b - mv) for a, b in zip(x, v)) / sum((a - mx) ** 2 for a in x)
    return [b - mv - b1 * (a - mx) for a, b in zip(x, v)]


def test_lm_lag_formula():
    e = _resid(Y)
    yhat = [a - b for a, b in zip(Y, e)]
    s2 = sum(v * v for v in e) / N
    T = sum(W[i][j] * W[i][j] + W[i][j] * W[j][i] for i in range(N) for j in range(N))
    Wyh = [sum(W[i][j] * yhat[j] for j in range(N)) for i in range(N)]
    MWyh = _resid(Wyh)
    nJ = sum(v * v for v in MWyh) / s2 + T
    eWy = sum(E_ * sum(W[i][j] * Y[j] for j in range(N)) for i, E_ in enumerate(e)) / s2
    eWe = sum(e[i] * W[i][j] * e[j] for i in range(N) for j in range(N)) / s2
    lm = eWy**2 / nJ
    r = sarsc(Y, X, W)
    assert abs(r.statistic - lm) < 1e-10
    assert abs(r.p_value - math.erfc(math.sqrt(lm / 2))) < 1e-10
    assert abs(r.extra["robust_statistic"] - (eWy - eWe) ** 2 / (nJ - T)) < 1e-10
