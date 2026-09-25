"""Tests for bsaclass.rangayyan_pca_signals (sec. 9.7.1)."""

import math

import pytest

from morie.fn.bsaclass import rangayyan_pca_signals


X = [[1, 2, 3, 4, 5, 6], [2, 4, 5, 8, 9, 13], [1, 0, 1, 0, 1, 0]]


def test_rgpca_basic():
    """Eigenpairs of the channel covariance: C v = lambda v, eigenvalues
    decreasing, their sum the total variance, and truncation MSE the sum
    of the discarded eigenvalues (eq. 9.40)."""
    r = rangayyan_pca_signals(X, ncomp=2)
    K, N = 3, 6
    m = [sum(row) / N for row in X]
    C = [[float(v) for v in row] for row in r["covariance"]]
    lam = [float(v) for v in r["eigenvalues"]]
    V = [[float(v) for v in row] for row in r["eigenvectors"]]
    assert lam == sorted(lam, reverse=True)
    for k in range(K):
        v = [V[i][k] for i in range(K)]
        Cv = [sum(C[i][j] * v[j] for j in range(K)) for i in range(K)]
        assert Cv == pytest.approx([lam[k] * t for t in v], abs=1e-9)
    assert sum(lam) == pytest.approx(sum(C[i][i] for i in range(K)), rel=1e-12)
    assert r["mse"] == pytest.approx(lam[2], abs=1e-12)
    # the covariance is that of the channels (rows)
    dev = [[X[a][t] - m[a] for t in range(N)] for a in range(K)]
    for den in (N - 1, N):
        ref = [[sum(dev[a][t] * dev[b][t] for t in range(N)) / den for b in range(K)] for a in range(K)]
        if all(abs(ref[a][b] - C[a][b]) < 1e-12 for a in range(K) for b in range(K)):
            break
    else:
        raise AssertionError("covariance is neither the n nor the n-1 estimator")


def test_rgpca_edge():
    """Keeping every component leaves zero error; one sample per channel
    raises."""
    assert rangayyan_pca_signals(X)["mse"] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        rangayyan_pca_signals([[1.0], [2.0]])
