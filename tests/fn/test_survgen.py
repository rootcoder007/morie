"""Tests for survgen.general_estimating_eq_surv (Cox with a cluster sandwich)."""

import math

import pytest


def _data(n=30):
    k = range(n)
    x1 = [math.sin(1.3 * i) for i in k]
    x2 = [math.cos(0.7 * i) for i in k]
    t = [round(math.exp(1 + 0.5 * a - 0.3 * b + 0.8 * math.sin(5.1 * i)), 3) + i * 1e-4
         for i, a, b in zip(k, x1, x2)]
    e = [0.0 if i % 4 == 0 else 1.0 for i in k]
    cl = [i // 3 for i in k]
    return t, e, [[a, b] for a, b in zip(x1, x2)], cl


def _inv2(M):
    d = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    return [[M[1][1] / d, -M[0][1] / d], [-M[1][0] / d, M[0][0] / d]]


def _cox_parts(b, t, e, Z):
    """Breslow partial-likelihood score, information and Lin-Wei (1989)
    score residuals r_i = d_i (Z_i - zbar(t_i))
    - sum_{j: d_j = 1, t_j <= t_i} w_i (Z_i - zbar(t_j)) / S0(t_j)."""
    n = len(t)
    w = [math.exp(z[0] * b[0] + z[1] * b[1]) for z in Z]
    U = [0.0, 0.0]
    I = [[0.0, 0.0], [0.0, 0.0]]
    res = [[0.0, 0.0] for _ in range(n)]
    for j in range(n):
        if not e[j]:
            continue
        R = [k for k in range(n) if t[k] >= t[j]]
        S0 = sum(w[k] for k in R)
        zb = [sum(w[k] * Z[k][c] for k in R) / S0 for c in range(2)]
        for c in range(2):
            U[c] += Z[j][c] - zb[c]
            for d in range(2):
                I[c][d] += sum(w[k] * Z[k][c] * Z[k][d] for k in R) / S0 - zb[c] * zb[d]
            res[j][c] += Z[j][c] - zb[c]
            for k in R:
                res[k][c] -= w[k] * (Z[k][c] - zb[c]) / S0
    return U, I, res


def _fit(t, e, Z):
    b = [0.0, 0.0]
    for _ in range(50):
        U, I, _ = _cox_parts(b, t, e, Z)
        V = _inv2(I)
        b = [b[0] + V[0][0] * U[0] + V[0][1] * U[1], b[1] + V[1][0] * U[0] + V[1][1] * U[1]]
    return b


def _sandwich(V, res, groups=None):
    if groups is not None:
        agg = {}
        for g, r in zip(groups, res):
            a = agg.setdefault(g, [0.0, 0.0])
            a[0] += r[0]
            a[1] += r[1]
        res = list(agg.values())
    M = [[sum(r[a] * r[c] for r in res) for c in range(2)] for a in range(2)]
    VM = [[sum(V[a][k] * M[k][c] for k in range(2)) for c in range(2)] for a in range(2)]
    return [[sum(VM[a][k] * V[k][c] for k in range(2)) for c in range(2)] for a in range(2)]

from morie.fn.survgen import general_estimating_eq_surv


def test_survgen_basic():
    """The coefficients solve the Breslow score equations (checked by an
    independent Newton fit); the model SE is the inverse information and
    the robust SE the cluster-aggregated Lin-Wei sandwich.  coxph gives
    beta -0.872244178602, 0.657963934894 on this data."""
    t, e, Z, cl = _data()
    b = _fit(t, e, Z)
    _, I, res = _cox_parts(b, t, e, Z)
    V = _inv2(I)
    Rc = _sandwich(V, res, cl)
    r = general_estimating_eq_surv(t, e, Z, cl)
    assert [float(v) for v in r["beta"]] == pytest.approx(b, abs=1e-10)
    assert [float(v) for v in r["beta"]] == pytest.approx([-0.872244178602, 0.657963934894], abs=1e-9)
    assert [float(v) for v in r["se_model"]] == pytest.approx([math.sqrt(V[0][0]), math.sqrt(V[1][1])], rel=1e-9)
    assert [float(v) for v in r["se_robust"]] == pytest.approx([math.sqrt(Rc[0][0]), math.sqrt(Rc[1][1])], rel=1e-9)
    assert r["variance_inflation"] == pytest.approx(math.sqrt(Rc[0][0] / V[0][0]), rel=1e-9)
    assert r["n_clusters"] == 10 and r["n_events"] == 22


def test_survgen_edge():
    """Fewer than two events and a mis-sized covariate matrix raise."""
    t, e, Z, _ = _data()
    with pytest.raises(ValueError):
        general_estimating_eq_surv(t, [1.0] + [0.0] * (len(t) - 1), Z)
    with pytest.raises(ValueError):
        general_estimating_eq_surv(t, e, Z[:-1])
