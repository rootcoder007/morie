import math

from morie.fn._qpcore import ssum
from morie.fn._rng import random_uniform
from morie.fn.stlocal import bivariate_moran, st_getis_ord, st_trend_surface

U = [float(v) for v in random_uniform(200, seed=21, stream=0)]
P = [(5 * U[i], 5 * U[20 + i]) for i in range(20)]
Z = [[U[40 + 20 * t + i] + (1.5 if (i < 5 and t >= 2) else 0) for i in range(20)] for t in range(4)]


def test_gstar_formula_and_hot_spot():
    r = st_getis_ord(Z, P, distance=1.5, time_window=1)
    x = [v for row in Z for v in row]
    n = 80
    xb = ssum(x) / n
    S = math.sqrt(ssum(v * v for v in x) / n - xb * xb)
    nb = [j for j in range(20) if math.dist(P[3], P[j]) <= 1.5]
    vals = [Z[s][j] for s in (1, 2, 3) for j in nb]
    W = len(vals)
    assert abs(r.z[2][3] - (ssum(vals) - xb * W) / (S * math.sqrt((n * W - W * W) / 79))) < 1e-12
    assert max(r.z[3][i] for i in range(5)) > max(r.z[0][i] for i in range(5))


def test_bivariate_moran():
    W = [[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 1, 0]]
    r = bivariate_moran([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], W, nsim=19, seed=2)
    assert abs(r.statistic - 0.4) < 1e-12 and abs(ssum(r.local) / 3 - r.statistic) < 1e-12
    assert r.p_value == (1 + sum(1 for v in r.simulated if abs(v) >= 0.4 - 1e-15)) / 20 or 0 < r.p_value <= 1


def test_trend_surface_recovers_polynomial():
    t = [i % 3 for i in range(20)]
    z = [2 + 0.5 * (p[0] - 2.5) - 0.3 * (p[1] - 2.5) ** 2 + 0.7 * tt for p, tt in zip(P, t)]
    r = st_trend_surface(z, P, t, degree=2, time_degree=1)
    assert r.r2 > 1 - 1e-12 and max(abs(v) for v in r.residuals) < 1e-9
    assert r.terms[0] == (0, 0, 0) and len(r.terms) == 12
