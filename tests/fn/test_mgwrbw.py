"""Tests for morie.fn.mgwrbw: each selected bandwidth minimises the one-covariate AICc of its partial residual."""

import math

from morie.fn.mgwrbw import mgwrbw
from morie.fn.mgwrfit import mgwrfit

N = 20
P = [(float(i % 5), float(i // 5)) for i in range(N)]
X = [[math.sin(i + 0.5), (0.3 * i) % 1.1 + 0.2] for i in range(N)]
Y = [1.0 + (1 + 0.2 * P[i][0]) * X[i][0] - X[i][1] + 0.1 * math.cos(3 * i) for i in range(N)]


def _aicc(x, v, bw):
    n = len(x)
    fit, tr = [], 0.0
    for i in range(n):
        w = [math.exp(-(math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]) ** 2) / (2 * bw * bw)) for j in range(n)]
        den = sum(w[j] * x[j] ** 2 for j in range(n))
        fit.append(x[i] * sum(w[j] * x[j] * v[j] for j in range(n)) / den)
        tr += x[i] ** 2 * w[i] / den
    rss = sum((a - b) ** 2 for a, b in zip(v, fit))
    return n * math.log(rss / n) + n * math.log(2 * math.pi) + n * (n + tr) / (n - 2 - tr)


def test_selected_bandwidths_are_criterion_minima():
    bws = mgwrbw(Y, X, P, kernel="gaussian")
    r = mgwrfit(Y, X, P, bandwidths=bws, kernel="gaussian")
    Xa = [[1.0] + x for x in X]
    lo, hi = 1.0, 5.0
    for k, bw in enumerate(bws):
        assert lo - 1e-9 <= bw <= hi + 1e-9
        x = [row[k] for row in Xa]
        v = [r["residuals"][j] + x[j] * r["betas"][j][k] for j in range(N)]
        c = _aicc(x, v, bw)
        for d in (1e-3, -1e-3):
            if lo < bw + d < hi:
                assert c <= _aicc(x, v, bw + d) + 1e-9
