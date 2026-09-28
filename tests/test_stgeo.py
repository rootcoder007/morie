import math

from morie.fn._qpcore import ssum
from morie.fn._rng import random_normal, random_uniform
from morie.fn.stcovar import st_model_variogram
from morie.fn.stgeo import (
    st_block_kriging,
    st_fit,
    st_kriging_diagnostics,
    st_leave_h_out,
    st_local_kriging,
    st_predict_grid,
    st_simulate,
    st_smoothness,
    st_universal_kriging,
)
from morie.fn.stkrig import _st_krige, st_covariance, st_kriging_cv

M = {
    "type": "productSum",
    "k": 0.5,
    "space": {"model": "Sph", "psill": 0.8, "range": 2.0, "nugget": 0.05},
    "time": {"model": "Exp", "psill": 0.6, "range": 1.5},
}
U = [float(v) for v in random_uniform(200, seed=41, stream=0)]
P = [(3 * U[i], 3 * U[6 + i]) for i in range(6)]
COORDS = [P[i] for t in range(5) for i in range(6)]
TIMES = [t for t in range(5) for _ in range(6)]
Z = [1 + U[21 + 6 * t + i] + 0.3 * t for t in range(5) for i in range(6)]


def test_st_fit_recovers_generating_models():
    h = [0.5, 1, 2, 3, 0.5, 1, 2, 3, 0, 0.5, 1, 2, 3]
    u = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    g = st_model_variogram(
        h,
        u,
        "productSum",
        space={"psill": 0.8, "model": "Exp", "range": 2.0, "nugget": 0.1},
        time={"psill": 0.6, "model": "Exp", "range": 1.5, "nugget": 0.0},
        k=0.4,
    )
    r = st_fit(h, u, g, [10] * 13, "productSum", [0.5, 1, 0.05, 0.5, 1, 0.05, 0.2])
    assert r.objective < 1e-18
    assert max(abs(a - b) for a, b in zip(r.par, [0.8, 2.0, 0.1, 0.6, 1.5, 0.0, 0.4])) < 1e-3
    g = st_model_variogram(
        h,
        u,
        "separable",
        space={"psill": 0.9, "model": "Exp", "range": 1.5, "nugget": 0.1},
        time={"psill": 1.0, "model": "Exp", "range": 2.0, "nugget": 0.0},
        sill=3.0,
    )
    r = st_fit(h, u, g, [10] * 13, "separable", [1, 0.2, 1, 0.2, 1], fit_method=1)
    assert max(abs(a - b) for a, b in zip(r.par, [1.5, 0.1, 2.0, 0.0, 3.0])) < 1e-5
    n = 13
    assert abs(r.aic - (n * math.log(r.objective) + 10)) < 1e-9


def test_universal_and_ordinary_kriging_agree_and_interpolate():
    Q, S = [(1, 1), (2.5, 0.4)], [1, 3]
    ok = _st_krige(Z, COORDS, TIMES, Q, S, M)
    uk = st_universal_kriging(Z, [[1.0]] * 30, COORDS, TIMES, [[1.0]] * 2, Q, S, M)
    assert max(abs(a - b) for a, b in zip(ok.prediction + ok.variance, uk.prediction + uk.variance)) < 1e-12
    X = [[1.0, t] for t in TIMES]
    r = st_universal_kriging(Z, X, COORDS, TIMES, [X[3]], [COORDS[3]], [TIMES[3]], M)
    assert abs(r.prediction[0] - Z[3]) < 1e-9 and abs(r.variance[0]) < 1e-9


def test_local_block_leave_h_out_grid_and_diagnostics():
    Q, S = [(1, 1)], [2]
    g = _st_krige(Z, COORDS, TIMES, Q, S, M)
    loc = st_local_kriging(Z, COORDS, TIMES, Q, S, M, nmax=30, stani=1.0)
    assert abs(loc.prediction[0] - g.prediction[0]) < 1e-12 and loc.neighbours[0] == list(range(30))
    small = st_local_kriging(Z, COORDS, TIMES, Q, S, M, nmax=5, stani=1.0)
    assert len(small.neighbours[0]) == 5
    pt = st_block_kriging(Z, COORDS, TIMES, Q, S, M, block=1.0, duration=1.0, n_space=1, n_time=1)
    assert abs(pt.prediction[0] - g.prediction[0]) < 1e-12
    bk = st_block_kriging(Z, COORDS, TIMES, Q, S, M, block=1.0, duration=1.0)
    assert bk.variance[0] < pt.variance[0] and bk.block_variance[0] < st_covariance(0, 0, M)
    cv = st_kriging_cv(Z, COORDS, TIMES, M)
    lh = st_leave_h_out(Z, COORDS, TIMES, M, h=0, tau=0)
    assert max(abs(a - b) for a, b in zip(cv.prediction, lh.prediction)) < 1e-12
    lh2 = st_leave_h_out(Z, COORDS, TIMES, M, h=0.5, tau=1)
    assert all(a <= 29 for a in lh2.n_used) and min(lh2.n_used) < 29
    d = st_kriging_diagnostics(COORDS, TIMES, Q, S, M)
    assert abs(ssum(w * z for w, z in zip(d.weights[0], Z)) - g.prediction[0]) < 1e-12
    assert (
        abs(d.weight_sums[0] - 1) < 1e-12
        and abs(d.relative_variance[0] - g.variance[0] / st_covariance(0, 0, M)) < 1e-12
    )
    grid = st_predict_grid(Z, COORDS, TIMES, M, [0.5, 1.5], [1.0, 2.0], [0, 4])
    ref = _st_krige(Z, COORDS, TIMES, [(1.5, 2.0)], [4], M)
    assert abs(grid.frames[1][1][1] - ref.prediction[0]) < 1e-12


def test_smoothness_and_simulation():
    assert st_smoothness(M).space == -1 and st_smoothness(M).time == 0
    s = st_smoothness({"type": "metric", "joint": {"model": "Mat", "kappa": 2.5}})
    assert (s.space, s.time) == (2, 2)
    Q = COORDS[:6]
    S = TIMES[:6]
    r = st_simulate(Q, S, M, nsim=2, seed=5, mean=1.0)
    L = r.cholesky
    C = [[st_covariance(math.dist(Q[i], Q[j]), S[i] - S[j], M) for j in range(6)] for i in range(6)]
    assert max(abs(ssum(L[i][k] * L[j][k] for k in range(6)) - C[i][j]) for i in range(6) for j in range(6)) < 1e-12
    e = [float(v) for v in random_normal(6, seed=5, stream=1)]
    assert max(abs(r.realisations[1][i] - 1 - ssum(L[i][k] * e[k] for k in range(6))) for i in range(6)) < 1e-12
    m2 = {"type": "metric", "stAni": 1.0, "joint": {"model": "Exp", "psill": 1.0, "range": 1.0}}
    c = st_simulate([(0, 0), (1, 0)], [0, 0], m2, z=[2.0], data_coords=[(0, 0)], data_times=[0])
    assert abs(c.realisations[0][0] - 2.0) < 1e-12
