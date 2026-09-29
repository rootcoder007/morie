"""Tests for micord.mi_pmm: donors are the nearest predicted means."""

import math

from morie.fn.micord import mi_pmm

X = [[math.sin(0.9 * i), 0.1 * i] for i in range(30)]
Y = [1.0 + 2.0 * r[0] - r[1] + 0.2 * math.cos(2.3 * i) for i, r in enumerate(X)]
R = [0 if i % 4 == 1 else 1 for i in range(30)]


def test_donors():
    r = mi_pmm(Y, X, R, K=3, seed=2)
    b = r["coefficients"]
    yhat = [b[0] + b[1] * x[0] + b[2] * x[1] for x in X]
    obs = [i for i in range(30) if R[i] == 1]
    for i, d in zip(r["missing"], r["donors"]):
        pool = sorted(obs, key=lambda j: (abs(yhat[i] - yhat[j]), j))[:3]
        assert d in pool
        assert r["imputed"][i] == Y[d]
    assert all(r["imputed"][i] == Y[i] for i in obs)
