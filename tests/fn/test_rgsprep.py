"""Tests for bsaclass.rangayyan_sparse_rep (OMP and lasso, sec. 9.5)."""

import math

import pytest

from morie.fn.bsaclass import rangayyan_sparse_rep


D = [[math.cos(0.37 * (k + 1) * j + 0.2 * k) for j in range(8)] for k in range(12)]
X = [2.0 * D[3][j] - 1.0 * D[7][j] + 0.01 * math.sin(3.1 * j) for j in range(8)]


def _recon(a):
    return [sum(a[k] * D[k][j] for k in range(12)) for j in range(8)]


def test_rgsprep_basic():
    """OMP (T = 2): at most two atoms, and the coefficients are the least
    squares fit on that support, so the residual is orthogonal to every
    selected atom.  Lasso: KKT, <d_k, x - D'a> = lam sign(a_k) on the
    support and |.| <= lam off it."""
    r = rangayyan_sparse_rep(X, D, sparsity=2)
    a = [float(v) for v in r["alpha"]]
    S = [k for k in range(12) if abs(a[k]) > 0]
    assert len(S) <= 2
    res = [x - y for x, y in zip(X, _recon(a))]
    for k in S:
        # 1e-9: the least-squares solve leaves rounding of order 1e-10
        assert sum(D[k][j] * res[j] for j in range(8)) == pytest.approx(0.0, abs=1e-9)
    lam = 0.05
    rl = rangayyan_sparse_rep(X, D, lam=lam, maxiter=20000, tol=1e-13)
    al = [float(v) for v in rl["alpha"]]
    rres = [x - y for x, y in zip(X, _recon(al))]
    for k in range(12):
        c = sum(D[k][j] * rres[j] for j in range(8))
        if abs(al[k]) > 1e-10:
            assert c == pytest.approx(lam * math.copysign(1, al[k]), abs=1e-6)
        else:
            assert abs(c) <= lam + 1e-6


def test_rgsprep_edge():
    """Exactly one of sparsity and lam must be given."""
    with pytest.raises(ValueError):
        rangayyan_sparse_rep(X, D)
    with pytest.raises(ValueError):
        rangayyan_sparse_rep(X, D, sparsity=2, lam=0.1)
