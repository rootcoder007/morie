"""Tests for snp_blup_gebv (SNP-BLUP, MVSML 2022 eq. 2.4)."""

import math

import pytest

from morie.fn.snp_blup_gebv import snp_blup_gebv


def _solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def _data(n=12, p=5):
    X = [[1.0, float(k % 2)] for k in range(n)]
    M = [[float((3 * i + 7 * j + i * j) % 3) - 1.0 for j in range(p)] for i in range(n)]
    y = [2.0 + 0.5 * x[1] + 0.3 * m[0] - 0.2 * m[3] + 0.1 * math.sin(5.0 * i)
         for i, (x, m) in enumerate(zip(X, M))]
    return X, y, M


def _mme(X, y, M, s2m, s2e):
    """Henderson's mixed model equations with Z = M, G = s2m I, R = s2e I:
    [X'X  X'M; M'X  M'M + (s2e/s2m) I] [b; u] = [X'y; M'y]."""
    C = [x + m for x, m in zip(X, M)]
    q, px = len(C[0]), len(X[0])
    A = [[sum(c[r] * c[s] for c in C) + (s2e / s2m if r == s and r >= px else 0.0)
          for s in range(q)] for r in range(q)]
    sol = _solve(A, [sum(c[r] * t for c, t in zip(C, y)) for r in range(q)])
    return sol[:px], sol[px:]


def test_msm243_basic():
    """Fixed effects, marker effects and GEBV = M u solve Henderson's MME."""
    X, y, M = _data()
    b, u = _mme(X, y, M, 0.4, 1.3)
    r = snp_blup_gebv(X, y, M, 0.4, 1.3)
    assert r["beta"] == pytest.approx(b, abs=1e-9)
    assert r["marker_effects"] == pytest.approx(u, abs=1e-9)
    gebv = [sum(a * c for a, c in zip(m, u)) for m in M]
    assert r["gebv"] == pytest.approx(gebv, abs=1e-9)
    assert r["estimate"] == pytest.approx(gebv[0], abs=1e-9)


def test_msm243_edge():
    """The book's equivalence: SNP-BLUP breeding values equal GBLUP's
    s2m M M' V^{-1} (y - X b) with V = s2m M M' + s2e I."""
    X, y, M = _data()
    s2m, s2e = 0.4, 1.3
    r = snp_blup_gebv(X, y, M, s2m, s2e)
    n = len(y)
    G = [[s2m * sum(a * c for a, c in zip(M[i], M[j])) for j in range(n)] for i in range(n)]
    V = [[G[i][j] + (s2e if i == j else 0.0) for j in range(n)] for i in range(n)]
    e = [t - sum(a * c for a, c in zip(x, r["beta"])) for x, t in zip(X, y)]
    w = _solve(V, e)
    g = [sum(G[i][j] * w[j] for j in range(n)) for i in range(n)]
    assert r["gebv"] == pytest.approx(g, abs=1e-9)
