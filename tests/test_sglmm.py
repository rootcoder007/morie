import math

from morie.fn._qpcore import ssum
from morie.fn._rng import random_normal, random_uniform
from morie.fn.sglmm import (
    _laplace,
    car_precision,
    crps_gaussian,
    crps_poisson,
    crps_sample,
    glmm_residuals,
    gmrf_simulate,
    sar_covariance,
    spatial_glmm_fit,
    spatial_glmm_predict,
    spatial_glmm_simulate,
)

U = [float(v) for v in random_uniform(200, seed=7, stream=0)]
Y = [
    5,
    6,
    0,
    1,
    6,
    6,
    0,
    1,
    1,
    0,
    2,
    3,
    1,
    1,
    6,
    1,
    4,
    6,
    0,
    2,
    2,
    0,
    4,
    2,
    14,
    7,
    2,
    8,
    3,
    6,
    12,
    8,
    2,
    7,
    8,
    5,
    2,
    4,
    6,
    7,
    7,
    17,
    2,
    1,
    2,
    5,
    0,
    1,
    0,
    3,
    1,
    4,
    0,
    7,
    5,
    8,
    5,
    2,
    6,
    1,
]
G = [i // 6 for i in range(60)]
X = [[1.0, U[i]] for i in range(60)]


def test_latent_fields():
    Q = car_precision([[0, 1, 0], [1, 0, 1], [0, 1, 0]], 0.4, 2.0)
    assert Q[1] == [-0.8, 4.0, -0.8]
    S = sar_covariance([[0, 1], [1, 0]], 0.5, 2.0)
    B = [[1, -0.5], [-0.5, 1]]
    BtB = [[ssum(B[k][i] * B[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    assert (
        max(
            abs(ssum(BtB[i][k] * S[k][j] for k in range(2)) - (2.0 if i == j else 0.0))
            for i in range(2)
            for j in range(2)
        )
        < 1e-12
    )
    r = gmrf_simulate(Q, nsim=2, seed=3)
    L = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(i + 1):
            s = Q[i][j] - ssum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = math.sqrt(s) if i == j else s / L[j][j]
    e = [float(v) for v in random_normal(3, seed=3, stream=1)]
    x = r.samples[1]
    assert max(abs(ssum(L[k][i] * x[k] for k in range(3)) - e[i]) for i in range(3)) < 1e-12


def test_simulation_families():
    lat = [0.1 * i for i in range(10)]
    Xs = [[1.0]] * 10
    for fam, kw in (
        ("poisson", {}),
        ("binomial", {"trials": 5}),
        ("negbin", {"size": 2.0}),
        ("gamma", {"shape": 3.0}),
        ("gaussian", {"sigma": 0.5}),
    ):
        r = spatial_glmm_simulate(Xs, [0.3], lat, family=fam, seed=4, **kw)
        assert len(r.y) == 10
        if fam in ("poisson", "negbin", "binomial"):
            assert all(v == int(v) and v >= 0 for v in r.y)
    p = spatial_glmm_simulate([[1.0]], [math.log(3.0)], [0.0], family="poisson", seed=9)
    u = float(random_uniform(1, seed=9, stream=0)[0])
    k, pk, F = 0, math.exp(-3.0), math.exp(-3.0)
    while u > F:
        k += 1
        pk *= 3.0 / k
        F += pk
    assert p.y[0] == k


def test_laplace_fit_is_stationary_and_matches_hand_objective():
    r = spatial_glmm_fit(Y, X, groups=G)
    Z = [[1.0 if g == k else 0.0 for k in range(10)] for g in G]
    S = [[r.sigma2 if a == b else 0.0 for b in range(10)] for a in range(10)]
    ll = _laplace(Y, X, Z, S, r.beta, "poisson", [1.0] * 60, [0.0] * 10)[0]
    assert abs(ll - r.loglik) < 1e-9
    h = 1e-5
    for k in range(2):
        bp = list(r.beta)
        bm = list(r.beta)
        bp[k] += h
        bm[k] -= h
        d = (
            _laplace(Y, X, Z, S, bp, "poisson", [1.0] * 60, [0.0] * 10)[0]
            - _laplace(Y, X, Z, S, bm, "poisson", [1.0] * 60, [0.0] * 10)[0]
        ) / (2 * h)
        assert abs(d) < 1e-4


def test_spatial_fit_predict_residuals_crps():
    P = [(4 * U[i], 4 * U[30 + i]) for i in range(20)]
    y = [3, 5, 2, 8, 4, 1, 6, 7, 2, 3, 9, 4, 5, 2, 1, 6, 3, 7, 4, 5]
    f = spatial_glmm_fit(y, [[1.0]] * 20, coords=P)
    assert f.range > 0 and f.sigma2 > 0 and len(f.posterior_sd) == 20
    p = spatial_glmm_predict(f, [[1.0]], [P[4]])
    assert abs(p.eta[0] - f.eta[4]) < 1e-6
    rq = glmm_residuals([2, 0, 5], [2.0, 1.0, 3.0], seed=2)
    assert rq.pearson[0] == 0.0 and abs(rq.pearson[1] + 1) < 1e-15
    assert abs(crps_gaussian([0.0], [0.0], [1.0])[0] - (2 / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi))) < 1e-12
    c = crps_poisson([2], [2.5])[0]
    Fk = [ssum(math.exp(-2.5) * 2.5**j / math.factorial(j) for j in range(k + 1)) for k in range(80)]
    assert abs(c - ssum((F - (1.0 if k >= 2 else 0.0)) ** 2 for k, F in enumerate(Fk))) < 1e-12
    assert crps_sample([1.0], [[0.0, 2.0]]) == [0.5]
