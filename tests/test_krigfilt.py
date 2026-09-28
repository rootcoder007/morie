import math

from morie.fn._qpcore import inverse, ssum
from morie.fn._rng import random_uniform
from morie.fn.krgsys import krige, kriging_covariance
from morie.fn.krigfilt import collocated_cokriging, filtered_krige, indicator_ccdf, kriging_efficiency

U = [float(v) for v in random_uniform(90, seed=31, stream=0)]
P = [(4 * U[i], 4 * U[30 + i]) for i in range(30)]
Z = [1 + math.sin(p[0]) + 0.5 * p[1] + 0.3 * U[60 + i] for i, p in enumerate(P)]
M = {"model": "Exp", "psill": 1.0, "range": 1.5}
Q = [(1.1, 2.2), (3.3, 0.7)]


def test_filtered_kriging_is_nugget_kriging_off_the_data():
    f = filtered_krige(Z, P, Q, M, 0.2)
    k = krige(Z, P, Q, [M, {"model": "Nug", "psill": 0.2}])
    assert max(abs(a - b) for a, b in zip(f.prediction, k.prediction)) < 1e-12
    assert max(abs(a - (b - 0.2)) for a, b in zip(f.variance, k.variance)) < 1e-12
    at = filtered_krige(Z, P, [P[3]], M, 0.2)
    assert abs(at.prediction[0] - Z[3]) > 1e-3
    exact = filtered_krige(Z, P, [P[3]], M, 0.0)
    assert abs(exact.prediction[0] - Z[3]) < 1e-10
    sk = filtered_krige(Z, P, Q, M, 0.2, beta=2.0)
    assert abs(sk.prediction[0] - (2 + ssum(w * (z - 2) for w, z in zip(sk.weights[0], Z)))) < 1e-12


def test_indicator_ccdf_order_relations_and_etype():
    T = [1.5, 2.5, 3.5]
    r = indicator_ccdf(Z, P, Q, T, M)
    for k, t in enumerate(T):
        raw = krige([1.0 if v <= t else 0.0 for v in Z], P, Q, M).prediction
        assert max(abs(r.raw[q][k] - raw[q]) for q in range(2)) < 1e-12
    for G, mu in zip(r.ccdf, r.etype):
        assert all(0 <= a <= b <= 1 for a, b in zip(G, G[1:]))
        assert min(Z) <= mu <= max(Z)


def test_kriging_efficiency_and_slope():
    r = kriging_efficiency(Z, P, Q, M)
    k = krige(Z, P, Q, M)
    for i in range(2):
        mu = abs(k.lagrange[i][0])
        assert abs(r.efficiency[i] - (1 - k.variance[i])) < 1e-12
        assert abs(r.slope[i] - (1 - k.variance[i] + mu) / (1 - k.variance[i] + 2 * mu)) < 1e-12


def test_collocated_cokriging_system():
    y0 = [0.4, -0.3]
    r0 = collocated_cokriging(Z, P, y0, Q, M, 0.0, mean_z=2.0)
    sk = krige(Z, P, Q, M, beta=2.0)
    assert max(abs(a - b) for a, b in zip(r0.prediction, sk.prediction)) < 1e-12
    r = collocated_cokriging(Z[:5], P[:5], [0.4], [Q[0]], M, 0.7, mean_z=2.0, mean_y=0.1, var_y=2.0)
    s = 0.7 * math.sqrt(2.0)
    C = [[kriging_covariance(math.dist(P[i], P[j]), M) for j in range(5)] for i in range(5)]
    c0 = [kriging_covariance(math.dist(P[i], Q[0]), M) for i in range(5)]
    A = [C[i] + [s * c0[i]] for i in range(5)] + [[s * v for v in c0] + [2.0]]
    b = c0 + [s]
    w = [ssum(float(inverse(A)[i][j]) * b[j] for j in range(6)) for i in range(6)]
    assert max(abs(a - c) for a, c in zip(r.weights[0] + [r.nu[0]], w)) < 1e-12
    assert abs(r.prediction[0] - (2 + ssum(w[i] * (Z[i] - 2) for i in range(5)) + w[5] * 0.3)) < 1e-12
