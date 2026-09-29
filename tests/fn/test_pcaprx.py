"""Tests for pcaprx: eigen-equations of the correlation matrix."""

import math

import pytest

from morie.fn.pcaprx import pcaprx

X = [[2.5, 2.4, 1.0], [0.5, 0.7, 2.1], [2.2, 2.9, 0.4], [1.9, 2.2, 1.8], [3.1, 3.0, 0.9], [2.3, 2.7, 1.1]]


def _corr():
    n, p = len(X), len(X[0])
    mu = [math.fsum(r[j] for r in X) / n for j in range(p)]
    sd = [math.sqrt(math.fsum((r[j] - mu[j]) ** 2 for r in X) / (n - 1)) for j in range(p)]
    Z = [[(r[j] - mu[j]) / sd[j] for j in range(p)] for r in X]
    R = [[math.fsum(z[a] * z[b] for z in Z) / (n - 1) for b in range(p)] for a in range(p)]
    return Z, R


def test_pcaprx_components_solve_the_eigen_equations():
    Z, R = _corr()
    r = pcaprx(X)
    V = [[float(v) for v in row] for row in r["components"]]
    lam = [float(v) for v in r["explained_variance"]]
    assert math.fsum(lam) == pytest.approx(3.0, rel=1e-12)  # trace of a correlation matrix
    assert lam == sorted(lam, reverse=True)
    for v, lv in zip(V, lam):
        Rv = [math.fsum(a * b for a, b in zip(row, v)) for row in R]
        assert Rv == pytest.approx([lv * t for t in v], abs=1e-10)
        assert math.fsum(t * t for t in v) == pytest.approx(1.0, rel=1e-12)
        assert max(v, key=abs) > 0
    S = [[float(t) for t in row] for row in r["scores"]]
    for i, z in enumerate(Z):
        assert S[i] == pytest.approx([math.fsum(a * b for a, b in zip(z, v)) for v in V], abs=1e-12)
    ratio = [float(t) for t in r["explained_variance_ratio"]]
    assert ratio == pytest.approx([v / 3.0 for v in lam], rel=1e-12)
    cum = [sum(ratio[: k + 1]) for k in range(3)]
    assert r["n_for_80"] == next(k + 1 for k, c in enumerate(cum) if c >= 0.8)
