"""Tests for morie.fn.scdisp: the Cameron-Trivedi statistic recomputed from the Poisson score equations."""

import math

from morie.fn.scdisp import scdisp

Y = [0, 3, 1, 9, 0, 6, 2, 1, 4, 12]
X = [[1.0, v] for v in (0.1, 0.4, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8, 0.6, 1.0)]


def _poisson_fit():
    # Newton on the two Poisson score equations
    b = [math.log(sum(Y) / len(Y)), 0.0]
    for _ in range(100):
        mu = [math.exp(b[0] + b[1] * r[1]) for r in X]
        g = [sum((y - m) * r[k] for y, m, r in zip(Y, mu, X)) for k in range(2)]
        H = [[sum(m * r[a] * r[c] for m, r in zip(mu, X)) for c in range(2)] for a in range(2)]
        d = H[0][0] * H[1][1] - H[0][1] ** 2
        b = [b[0] + (H[1][1] * g[0] - H[0][1] * g[1]) / d, b[1] + (H[0][0] * g[1] - H[0][1] * g[0]) / d]
    return [math.exp(b[0] + b[1] * r[1]) for r in X]


def test_constant_dispersion_form():
    mu = _poisson_fit()
    n = len(Y)
    a = [((y - m) ** 2 - y) / m for y, m in zip(Y, mu)]
    al = sum(a) / n
    se = math.sqrt(sum((v - al) ** 2 for v in a) / (n - 1) / n)
    r = scdisp(Y, X)
    assert abs(r.extra["alpha"] - al) < 1e-9
    assert abs(r.statistic - al / se) < 1e-8
    assert abs(r.p_value - 0.5 * math.erfc(al / se / math.sqrt(2))) < 1e-9
    assert abs(r.extra["dispersion"] - (1 + al)) < 1e-9


def test_nb2_form():
    mu = _poisson_fit()
    a = [((y - m) ** 2 - y) / m for y, m in zip(Y, mu)]
    smm = sum(m * m for m in mu)
    al = sum(u * m for u, m in zip(a, mu)) / smm
    se = math.sqrt(sum((u - al * m) ** 2 for u, m in zip(a, mu)) / (len(Y) - 1) / smm)
    r = scdisp(Y, X, trafo=2)
    assert abs(r.statistic - al / se) < 1e-8
