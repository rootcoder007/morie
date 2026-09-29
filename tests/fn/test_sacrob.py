"""Tests for morie.fn.sacrob: every expected value is recomputed from the formula."""

import math

from morie.fn.sacrob import sacrob

N = 8
_B = [[1.0 if abs(i - j) == 1 or {i, j} == {0, 7} or {i, j} == {2, 5} else 0.0 for j in range(N)] for i in range(N)]
W = [[v / sum(r) for v in r] for r in _B]
X = [[1.0, ((i * 7) % 11) / 5] for i in range(N)]
Y = [1 + 2 * X[i][1] + ((i * 3) % 5 - 2) / 4 + 0.3 * sum(W[i][j] * X[j][1] for j in range(N)) for i in range(N)]
E = [((i * 4) % 7 - 3) / 5 for i in range(N)]
W3 = [[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]]


def test_hc0_sandwich_of_the_final_gs2sls_step():
    from morie.fn.gmsar import gs2sls_sac

    r = sacrob(Y, X, W)
    g = gs2sls_sac(Y, X, W, robust="HC0").value
    assert r.statistic == g["rho"]
    assert r.extra["se"] == list(g["se"])
    # the HC0 covariance differs from the homoskedastic one
    h = gs2sls_sac(Y, X, W).value
    assert max(abs(a - b) for a, b in zip(r.extra["se"], h["se"])) > 1e-6


def test_hc1_scales_up():
    r0, r1 = sacrob(Y, X, W), sacrob(Y, X, W, robust="HC1")
    f = math.sqrt(N / (N - 3))  # n / (n - k) with k = 3 (rho, intercept, slope)
    assert all(abs(b - f * a) < 1e-12 for a, b in zip(r0.extra["se"], r1.extra["se"]))
