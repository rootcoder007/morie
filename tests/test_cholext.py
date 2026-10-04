import math

from morie.fn._qpcore import ssum
from morie.fn._rng import random_normal
from morie.fn.cholext import (
    banded_cholesky,
    block_lu_simulate,
    incomplete_cholesky,
    lmc_cosimulate,
    nested_decomposition,
    refined_solve,
    simulation_sensitivity,
    tapered_simulate,
    wendland_taper,
)
from morie.fn.krgsys import kriging_covariance

SPH = {"model": "Sph", "psill": 1.0, "range": 2.5}


def _full_chol(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - ssum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = math.sqrt(s) if i == j else s / L[j][j]
    return L


def test_banded_and_incomplete_cholesky():
    A = [[kriging_covariance(abs(i - j), SPH) + (0.1 if i == j else 0) for j in range(10)] for i in range(10)]
    b = banded_cholesky(A, 2)
    F = _full_chol(A)
    assert b.ignored_max == 0 and max(abs(b.L[i][j] - F[i][j]) for i in range(10) for j in range(10)) < 1e-14
    ic = incomplete_cholesky(A)
    assert ic.relative_error < 1e-14
    G = [[math.exp(-abs(i - j) / 3.0) for j in range(8)] for i in range(8)]
    ic2 = incomplete_cholesky(G, drop_tol=0.3)
    assert 0 < ic2.relative_error < 0.5 and ic2.nnz < 36


def test_wendland_and_tapered_simulation():
    for k in range(4):
        v = wendland_taper([0.0, 0.3, 1.2], 1.0, dimension=2, k=k)
        assert v[0] == 1.0 and v[2] == 0.0 and 0 < v[1] < 1
    l_ = 4
    assert (
        abs(
            wendland_taper([0.3], 1.0, dimension=2, k=2)[0]
            - 0.7**6 * ((l_ * l_ + 4 * l_ + 3) * 0.09 + (3 * l_ + 6) * 0.3 + 3) / 3
        )
        < 1e-15
    )
    P = [(i, 0.5 * i) for i in range(8)]
    r = tapered_simulate(P, {"model": "Exp", "psill": 1.0, "range": 2.0}, 3.0, nsim=2, seed=4)
    L = _full_chol(r.cov)
    e = [float(v) for v in random_normal(8, seed=4, stream=1)]
    assert max(abs(r.simulations[1][i] - ssum(L[i][t] * e[t] for t in range(i + 1))) for i in range(8)) < 1e-12
    assert r.sparsity > 0.3


def test_lmc_sensitivity_nested():
    comps = [
        ([[1.0, 0.6], [0.6, 2.0]], {"model": "Exp", "psill": 1.0, "range": 1.0}),
        ([[0.5, 0.0], [0.0, 0.3]], {"model": "Nug", "psill": 1.0}),
    ]
    P = [(0, 0), (1, 0), (0, 2)]
    r = lmc_cosimulate(P, comps)
    assert abs(r.cov[0][3] - (0.6 + 0.0)) < 1e-15 and abs(r.cov[4][4] - 2.3) < 1e-15
    m = {"model": "Exp", "psill": 1.0, "range": 1.0}
    s = simulation_sensitivity(P, m, sills=[1.0, 4.0])
    assert all(abs(b - 2 * a) < 1e-15 for a, b in zip(*s.simulations))
    g = simulation_sensitivity(P, m, ranges=[1.0, 1.5, 5.0])
    assert g.correlation[0] == 1.0 or abs(g.correlation[0] - 1) < 1e-15
    nd = nested_decomposition(P, [{"model": "Nug", "psill": 0.5}, m])
    assert max(abs(t - a - b) for t, a, b in zip(nd.total, *nd.components)) < 1e-15
    assert nd.nominal_share == [1 / 3, 2 / 3]


def test_refined_solve_and_block_simulation():
    G = [[math.exp(-(((i - j) / 3.0) ** 2)) + (1e-8 if i == j else 0) for j in range(12)] for i in range(12)]
    b = [math.sin(i) for i in range(12)]
    r = refined_solve(G, b, iterations=3)
    na = max(ssum(abs(v) for v in row) for row in G)
    nx = max(abs(v) for v in r.x)
    assert r.residual_norms[-1] <= 100 * 2.220446049250313e-16 * na * nx * 12
    m = {"model": "Exp", "psill": 1.0, "range": 1.0}
    one = block_lu_simulate([0, 1, 2], [0, 1], m, block=3, seed=2)
    pts = [(x, y) for y in (0, 1) for x in (0, 1, 2)]
    L = _full_chol([[kriging_covariance(math.dist(p, q), m) for q in pts] for p in pts])
    e = [float(v) for v in random_normal(6, seed=2, stream=0)]
    flat = [v for row in one.field for v in row]
    assert max(abs(flat[i] - ssum(L[i][t] * e[t] for t in range(i + 1))) for i in range(6)) < 1e-12
    ind = block_lu_simulate([0, 1, 2, 3], [0, 1], m, block=2, halo=0.0, seed=2)
    assert ind.n_blocks == 2
