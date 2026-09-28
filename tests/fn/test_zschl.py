"""Tests for zschl: Cholesky (LU) Gaussian random field simulation."""

import math

import pytest

from morie.fn._mvcore import cholesky
from morie.fn._rng import random_normal
from morie.fn.krgsys import _dist, krige, kriging_covariance
from morie.fn.zschl import chol, chol_sim, cholsim, pivoted_cholesky

C6 = [(0.3 * i, (0.7 * i) % 2.1) for i in range(6)]
M = [{"model": "Nug", "psill": 0.1}, {"model": "Exp", "psill": 1.0, "range": 0.8}]


def _cov(P, m):
    return [[kriging_covariance(_dist(a, b), m) for b in P] for a in P]


def test_unconditional_is_mean_plus_L_e():
    r = chol_sim(C6, M, nsim=2, seed=7, mean=2.0)
    L = cholesky(_cov(C6, M))
    for s in range(2):
        e = [float(v) for v in random_normal(6, seed=7, stream=s)]
        want = [2.0 + sum(L[i][k] * e[k] for k in range(6)) for i in range(6)]
        assert r.simulations[s] == pytest.approx(want, abs=1e-14)


def test_docstring_example_and_aliases():
    r = chol_sim([(0.0, 0.0), (1.0, 0.0)], {"model": "Exp", "psill": 1.0, "range": 1.0}, seed=3)
    assert [round(v, 6) for v in r.simulations[0]] == [0.902691, -0.775404]
    assert chol is chol_sim and cholsim is chol_sim


def test_precision_method_solves_R_transpose():
    r = chol_sim(C6, M, seed=4, method="precision")
    from morie.fn._qpcore import inverse

    R = cholesky([[float(v) for v in row] for row in inverse(_cov(C6, M))])
    e = [float(v) for v in random_normal(6, seed=4, stream=0)]
    x = r.simulations[0]
    # R' x = e with R lower triangular
    assert [sum(R[k][i] * x[k] for k in range(6)) for i in range(6)] == pytest.approx(e, abs=1e-12)


def test_precision_draws_have_the_model_covariance():
    P = [(0.0, 0.0), (0.5, 0.0)]
    m = {"model": "Exp", "psill": 1.0, "range": 1.0}
    n = 3000
    s = chol_sim(P, m, nsim=n, seed=11, method="precision").simulations
    c = sum(a * b for a, b in s) / n
    se = math.sqrt(sum((a * b - c) ** 2 for a, b in s) / (n - 1) / n)
    assert abs(c - math.exp(-0.5)) < 4.0 * se


def test_conditional_law_is_simple_kriging():
    D = [(0.1, 0.2), (1.0, 1.0), (1.4, 0.3)]
    z = [1.0, 0.5, 2.0]
    r = chol_sim(C6, M, nsim=2, seed=5, z=z, data_coords=D, mean=1.0)
    k = krige(z, D, C6, M, beta=1.0)
    assert r.mean == pytest.approx(k.prediction, abs=1e-12)
    assert [r.cov[i][i] for i in range(6)] == pytest.approx(k.variance, abs=1e-12)
    # a simulation location on a datum reproduces it
    s = chol_sim([(1.0, 1.0), (0.5, 0.5)], M, seed=2, z=z, data_coords=D)
    assert s.simulations[0][0] == pytest.approx(0.5, abs=1e-6)


def test_pivoted_cholesky_low_rank_and_reconstruction():
    P = [(0.05 * i, 0.0) for i in range(20)]
    A = _cov(P, {"model": "Gau", "psill": 1.0, "range": 2.0})
    r = pivoted_cholesky(A, tol=1e-10)
    assert r.rank < 20
    L = r.L
    err = max(abs(A[i][j] - sum(L[i][k] * L[j][k] for k in range(r.rank))) for i in range(20) for j in range(20))
    assert err < 1e-9 * 20
    assert r.pivots[0] == 0
    f = pivoted_cholesky([[4.0, 2.0], [2.0, 1.0]])
    assert (f.rank, f.pivots, [v[0] for v in f.L]) == (1, [0], [2.0, 1.0])
    s = chol_sim(P, {"model": "Gau", "psill": 1.0, "range": 2.0}, seed=1, method="pivoted", tol=1e-10)
    assert len(s.simulations[0]) == 20


def test_validation():
    with pytest.raises(ValueError):
        chol_sim(C6, M, method="svd")
    with pytest.raises(ValueError):
        chol_sim(C6, M, z=[1.0])
    with pytest.raises(ValueError):
        chol_sim(C6, M, z=[1.0], data_coords=[(0, 0), (1, 1)])
