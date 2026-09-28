import math

from morie.fn._qpcore import inverse
from morie.fn._rng import random_uniform
from morie.fn._s03core import chol
from morie.fn.krgsys import krige, kriging_covariance
from morie.fn.krigapprox import fixed_rank_kriging, nngp_predict, sparse_gp_krige, tapered_kriging, vecchia_loglik

M = {"model": "Exp", "psill": 1.2, "range": 1.3, "nugget": 0.15}


def _data(n=20):
    u = [float(v) for v in random_uniform(3 * n, seed=8)]
    P = [(4 * u[2 * i], 4 * u[2 * i + 1]) for i in range(n)]
    z = [P[i][0] - 0.5 * P[i][1] + u[2 * n + i] for i in range(n)]
    return P, z


def _C(A, B, model):
    return [[kriging_covariance(math.dist(a, b), model) for b in B] for a in A]


def test_vecchia_full_conditioning_is_exact():
    P, z = _data()
    L = chol(_C(P, P, M))
    n = len(z)
    y = []  # solve L y = z - mu
    for i in range(n):
        y.append((z[i] - 1.0 - sum(L[i][k] * y[k] for k in range(i))) / L[i][i])
    exact = -0.5 * n * math.log(2 * math.pi) - sum(math.log(L[i][i]) for i in range(n)) - 0.5 * sum(v * v for v in y)
    assert abs(vecchia_loglik(z, P, M, n - 1, mean=1.0).loglik - exact) < 1e-9
    assert vecchia_loglik(z, P, M, 3, mean=1.0).loglik != exact


def test_nngp_and_taper_reduce_to_simple_kriging():
    P, z = _data()
    Q = [(1.0, 1.0), (2.5, 3.1)]
    ref = krige(z, P, Q, M, beta=0.7)
    r = nngp_predict(z, P, Q, M, 20, mean=0.7)
    t = tapered_kriging(z, P, Q, M, 1e9, mean=0.7)
    for a, b, c in zip(r.prediction + r.variance, t.prediction + t.variance, ref.prediction + ref.variance):
        assert abs(a - c) < 1e-10 and abs(b - c) < 1e-8


def test_sparse_gp_with_all_knots_is_exact():
    P, z = _data(12)
    sig = {"model": "Exp", "psill": 1.2, "range": 1.3}
    Q = [(1.0, 1.0), (3.0, 0.5)]
    K = _C(P, P, sig)
    Ki = inverse([[K[i][j] + (0.15 if i == j else 0.0) for j in range(12)] for i in range(12)])
    for method in ("dtc", "fitc"):
        r = sparse_gp_krige(z, P, Q, sig, P, noise=0.15, method=method, mean=1.0)
        for q, pr, vr in zip(Q, r.prediction, r.variance):
            k = [kriging_covariance(math.dist(p, q), sig) for p in P]
            w = [sum(Ki[i][j] * k[j] for j in range(12)) for i in range(12)]
            assert abs(pr - (1.0 + sum(w[i] * (z[i] - 1.0) for i in range(12)))) < 1e-9
            assert abs(vr - (1.2 - sum(w[i] * k[i] for i in range(12)))) < 1e-9


def test_frk_woodbury_equals_dense():
    P, z = _data(15)
    U = [(0.5, 0.5), (3.5, 3.5), (2.0, 2.0)]
    K = [[1.0, 0.2, 0.3], [0.2, 1.0, 0.3], [0.3, 0.3, 1.5]]
    r = fixed_rank_kriging(z, P, [(1.0, 2.0)], U, 2.5, K, sigma2_eps=0.1, sigma2_xi=0.05, mean=1.0)

    def b(q):
        return [(1 - (math.dist(q, c) / 2.5) ** 2) ** 2 if math.dist(q, c) < 2.5 else 0.0 for c in U]

    S = [b(p) for p in P]
    s0 = b((1.0, 2.0))
    SK = [[sum(S[i][a] * K[a][c] for a in range(3)) for c in range(3)] for i in range(15)]
    Sig = [
        [sum(SK[i][c] * S[j][c] for c in range(3)) + (0.15 if i == j else 0.0) for j in range(15)] for i in range(15)
    ]
    Si = inverse(Sig)
    k0 = [sum(SK[i][c] * s0[c] for c in range(3)) for i in range(15)]
    w = [sum(Si[i][j] * k0[j] for j in range(15)) for i in range(15)]
    assert abs(r.prediction[0] - (1.0 + sum(w[i] * (z[i] - 1.0) for i in range(15)))) < 1e-10
    sks = sum(s0[a] * K[a][c] * s0[c] for a in range(3) for c in range(3))
    assert abs(r.mspe[0] - (sks + 0.05 - sum(w[i] * k0[i] for i in range(15)))) < 1e-10
