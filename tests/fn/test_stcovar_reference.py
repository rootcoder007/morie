"""stcovar: variogram and covariance formulas recomputed; gstat-exact values checked in tests/cross."""

import math

import pytest

from morie.fn.stcovar import st_covariance_family, st_linear_combination, st_model_variogram

S = {"psill": 2.0, "model": "Exp", "range": 100.0, "nugget": 0.5}
T = {"psill": 3.0, "model": "Sph", "range": 5.0}
J = {"psill": 1.5, "model": "Gau", "range": 40.0}


def test_gstat_parametrised_variograms():
    gs = 0.5 + 2 * (1 - math.exp(-0.5))
    gt = 3 * (1.5 * 0.6 - 0.5 * 0.216)
    assert st_model_variogram([50], [3], "productSum", space=S, time=T, k=0.2) == pytest.approx(
        [(0.2 * 3 + 1) * gs + (0.2 * 2.5 + 1) * gt - 0.2 * gs * gt], abs=1e-14
    )
    us = {"psill": 0.8, "model": "Exp", "range": 100.0, "nugget": 0.2}
    ut = {"psill": 1.0, "model": "Sph", "range": 5.0}
    g1, g2 = 0.2 + 0.8 * (1 - math.exp(-0.5)), 1.5 * 0.6 - 0.5 * 0.216
    assert st_model_variogram([50], [3], "separable", space=us, time=ut, sill=4.0) == pytest.approx(
        [4 * (g1 + g2 - g1 * g2)]
    )
    d = math.hypot(30, 10 * 2)
    gj = 1.5 * (1 - math.exp(-((d / 40) ** 2)))
    assert st_model_variogram([30], [2], "metric", joint=J, stani=10.0) == pytest.approx([gj], abs=1e-14)
    sm = st_model_variogram([30], [2], "sumMetric", space=S, time=T, joint=J, stani=10.0)[0]
    assert sm == pytest.approx(0.5 + 2 * (1 - math.exp(-0.3)) + 3 * (1.5 * 0.4 - 0.5 * 0.064) + gj, abs=1e-14)
    assert st_model_variogram([0], [0], "sumMetric", space=S, time=T, joint=J, stani=10.0) == [0.0]
    assert st_model_variogram([500], [50], "productSum", space=S, time=T, k=0.0)[0] == pytest.approx(
        0.5 + 2 * (1 - math.exp(-5)) + 3
    )
    with pytest.raises(ValueError):
        st_model_variogram([1], [1], "bogus")


def test_covariance_families():
    g = st_covariance_family(
        [2.0], [1.5], "gneiting", sigma2=2.0, a=0.5, alpha=0.8, beta=0.6, gamma=0.7, c=0.9, tau=1.0
    )[0]
    psi = 0.5 * 1.5**1.6 + 1
    assert g == pytest.approx(2 / psi * math.exp(-0.9 * 2**1.4 / psi ** (0.6 * 0.7)), abs=1e-15)
    sep = st_covariance_family([2.0], [1.5], "gneiting", a=0.5, alpha=0.8, beta=0.0, gamma=0.7, c=0.9, tau=1.0)[0]
    assert sep == pytest.approx(math.exp(-0.9 * 2**1.4) / psi)
    ch = st_covariance_family([1.2], [0.7], "cressie_huang", a=2.0, b=0.5, d=3)[0]
    q = 4 * 0.49 + 1
    assert ch == pytest.approx(q**-1.5 * math.exp(-0.25 * 1.44 / q), abs=1e-15)
    ic = st_covariance_family([3.0], [-2.0], "iaco_cesare", sigma2=1.5, a=2.0, b=4.0, alpha=1.5, beta=1.0, delta=2.0)[0]
    assert ic == pytest.approx(1.5 * (1 + 1.5**1.5 + 0.5) ** -2)
    pe = st_covariance_family([1.0, 1.0], [6.0, 12.0], "periodic", range=2.0, period=12.0)
    assert pe == pytest.approx([-math.exp(-0.5), math.exp(-0.5)])
    assert st_covariance_family([0.0], [0.0], "iaco_cesare", a=1.0, b=1.0, alpha=1.0, beta=1.0, delta=1.0) == [1.0]
    r = st_linear_combination(
        [1.0],
        [2.0],
        [("separable_exp", {"range_s": 2.0, "range_t": 4.0}), ("cressie_huang", {"a": 1.0, "b": 1.0})],
        [0.25, 0.75],
    )
    assert r.covariance[0] == pytest.approx(0.25 * math.exp(-1) + 0.75 * math.exp(-1 / 5) / 5)
    with pytest.raises(ValueError):
        st_linear_combination([1.0], [1.0], [("separable_exp", {"range_s": 1.0, "range_t": 1.0})], [-1.0])
    with pytest.raises(ValueError):
        st_covariance_family([1.0], [1.0], "bogus")


def test_covariance_matrix_is_positive_semidefinite():
    pts = [(0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 2.0, 0.5), (1.5, 1.5, 2.0), (0.3, 0.9, 3.0)]
    for fam, p in [
        ("gneiting", dict(a=1.0, alpha=0.9, beta=1.0, gamma=0.8, c=1.0, tau=1.0)),
        ("iaco_cesare", dict(a=1.0, b=1.0, alpha=1.8, beta=1.2, delta=0.7)),
    ]:
        M = [[st_covariance_family([math.dist(a[:2], b[:2])], [a[2] - b[2]], fam, **p)[0] for b in pts] for a in pts]
        # Cholesky succeeds only for a positive-definite matrix
        n = len(M)
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = M[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    assert s > 0
                    L[i][i] = math.sqrt(s)
                else:
                    L[i][j] = s / L[j][j]
