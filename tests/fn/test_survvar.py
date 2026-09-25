"""Tests for survvar.variance_cox_estimator (model and Lin-Wei robust variance)."""

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

from morie.fn.survvar import variance_cox_estimator


def test_survvar_basic():
    """Information-based and sandwich variances at the Breslow MLE equal
    the independent recomputation; on this data survival::coxph(ties =
    "breslow", robust = TRUE) gives robust SEs 0.367866762592 and
    0.297381920115, and with cluster(cl) 0.442759042763 and
    0.342087883757."""
    t, e, Z, cl = _data()
    b = _fit(t, e, Z)
    _, I, res = _cox_parts(b, t, e, Z)
    V = _inv2(I)
    r = variance_cox_estimator(b, Z, t, e, robust=True)
    assert r["se"] == pytest.approx([math.sqrt(V[0][0]), math.sqrt(V[1][1])], rel=1e-10)
    R = _sandwich(V, res)
    assert r["robust_se"] == pytest.approx([math.sqrt(R[0][0]), math.sqrt(R[1][1])], rel=1e-10)
    assert r["robust_se"] == pytest.approx([0.367866762592, 0.297381920115], abs=1e-9)
    rc = variance_cox_estimator(b, Z, t, e, robust=True, cluster=cl)
    Rc = _sandwich(V, res, cl)
    assert rc["robust_se"] == pytest.approx([math.sqrt(Rc[0][0]), math.sqrt(Rc[1][1])], rel=1e-10)
    assert rc["robust_se"] == pytest.approx([0.442759042763, 0.342087883757], abs=1e-9)


def test_survvar_edge():
    """Without robust the sandwich keys are None; bad shapes, non-binary
    events and no events raise."""
    t, e, Z, _ = _data()
    r = variance_cox_estimator([0.0, 0.0], Z, t, e)
    assert r["robust_se"] is None and r["ratio"] is None
    with pytest.raises(ValueError):
        variance_cox_estimator([0.0], Z, t, e)
    with pytest.raises(ValueError):
        variance_cox_estimator([0.0, 0.0], Z, t, [2.0] * len(t))
    with pytest.raises(ValueError):
        variance_cox_estimator([0.0, 0.0], Z, t, [0.0] * len(t))
