import math

from morie.fn._qpcore import ssum
from morie.fn._rng import random_uniform
from morie.fn.krgsys import krige
from morie.fn.simdiag import (
    boundary_probability,
    class_proportions,
    conditional_turning_bands,
    connectivity,
    directional_variogram,
    indicator_variogram,
    standardise_realisations,
    tb_band_convergence,
    tb_ensemble,
)
from morie.fn.zstbs import turning_bands

U = [float(v) for v in random_uniform(300, seed=33, stream=0)]


def test_conditional_turning_bands_honours_data_and_recomputes():
    D = [(0.0, 0.0), (2.0, 1.0), (1.0, 3.0)]
    Q = [(0.0, 0.0), (1.0, 1.0), (3.0, 3.0)]
    z = [1.5, 0.2, -0.4]
    r = conditional_turning_bands(z, D, Q, range_=2.0, seed=4)
    assert abs(r.field[0] - 1.5) < 1e-9
    f = [float(v) for v in turning_bands(D + Q, "exponential", range_=2.0, seed=4)["field"]]
    m = {"model": "Exp", "psill": 1.0, "range": 2.0}
    kz = krige(z, D, Q, m, beta=0.0)["prediction"]
    ks = krige(f[:3], D, Q, m, beta=0.0)["prediction"]
    assert max(abs(r.field[k] - (kz[k] + f[3 + k] - ks[k])) for k in range(3)) < 1e-12


def test_ensemble_convergence_and_standardise():
    P = [(0.5 * i, 0.0) for i in range(6)]
    e = tb_ensemble(P, nsim=8, range_=2.0, bounds=[0, 1, 2])
    assert len(e.variogram) == 2 and abs(e.model_variogram[0] - (1 - math.exp(-(6.5 / 9) / 2))) < 1e-12
    c = tb_band_convergence([(0, 0), (1, 0)], bands=(2, 32), nsim=30)
    assert len(c.rmse) == 2 and all(v >= 0 for v in c.rmse)
    s = standardise_realisations([[1.0, 2.0, 6.0]], mean=5.0, variance=4.0).realisations[0]
    m = ssum(s) / 3
    assert abs(m - 5) < 1e-12 and abs(ssum((v - m) ** 2 for v in s) / 2 - 4) < 1e-12


def test_directional_variogram_axes():
    P = [(i, j) for j in range(4) for i in range(4)]
    z = [float(i) for i, j in P]
    r = directional_variogram(z, P, [0, 90], [0, 1.5], tol=10)
    assert r.gamma[0] == [0.0] and r.gamma[1] == [0.5] and r.np[0] == [12]


def test_categorical_diagnostics():
    reals = [[1 if U[40 * r + k] < 0.55 else 0 for k in range(25)] for r in range(6)]
    cp = class_proportions(reals, [0, 1], target=[0.45, 0.55])
    assert abs(ssum(cp.mean) - 1) < 1e-15
    bp = boundary_probability(reals, 5, 5, [0, 1])
    assert all(0 <= v <= 1 for row in bp.boundary for v in row)
    cn = connectivity(reals, 5, 5, 1, pairs=[(0, 24)])
    assert all(n >= 1 for n in cn.n_components)
    full = connectivity([[1] * 25], 5, 5, 1, pairs=[(0, 24)])
    assert full.n_components == [1] and full.pair_probability == [1.0] and full.percolates == [True]
    iv = indicator_variogram([[1, 0, 1, 0]], 1, 4, 1, [1, 2])
    assert iv.x == [[0.5, 0.0]]
