"""Tests for vargpc.variational_gp_classifier (SVGP, Hensman et al. 2015)."""

import math

import pytest

from morie.fn.vargpc import variational_gp_classifier


X = [[-2.0 + 0.25 * i] for i in range(16)]
Y = [1 if x[0] > 0.1 else 0 for x in X]


def _inv_logdet(A):
    n = len(A)
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]
    ld = 0.0
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        ld += math.log(abs(pv))
        M[c] = [v / pv for v in M[c]]
        for r in range(n):
            if r != c:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [row[n:] for row in M], ld


def test_vargpc_basic():
    """KL(q(u) || p(u)) with q = N(m, S), p = N(0, K_mm) (RBF kernel,
    jitter on the diagonal) equals 1/2 [tr(K^-1 S) + m'K^-1 m - M
    + log|K| - log|S|], recomputed from the returned m, S and Z; the
    ELBO never decreases; a separable problem is fitted perfectly."""
    r = variational_gp_classifier(X, Y, m_inducing=4, lengthscale=1.0, variance=1.0, steps=60)
    Z = [[float(v) for v in z] for z in r["Z"]]
    M = len(Z)
    K = [[math.exp(-0.5 * sum((a - b) ** 2 for a, b in zip(Z[i], Z[j]))) + (1e-8 if i == j else 0.0)
          for j in range(M)] for i in range(M)]
    Ki, ldK = _inv_logdet(K)
    S = [[float(v) for v in row] for row in r["S"]]
    m = [float(v) for v in r["m"]]
    _, ldS = _inv_logdet(S)
    tr = sum(Ki[i][j] * S[j][i] for i in range(M) for j in range(M))
    quad = sum(m[i] * Ki[i][j] * m[j] for i in range(M) for j in range(M))
    assert r["kl"] == pytest.approx(0.5 * (tr + quad - M + ldK - ldS), rel=1e-10)
    assert r["elbo_monotone"] in (True, 1, 1.0)
    assert list(r["fit_pred"]) == Y
    assert all(0.0 < float(p) < 1.0 for p in r["fit_prob"])


def test_vargpc_edge():
    """Non-binary labels, M outside 1..n and non-positive
    hyperparameters raise."""
    with pytest.raises(ValueError):
        variational_gp_classifier(X, [2] * 16)
    with pytest.raises(ValueError):
        variational_gp_classifier(X, Y, m_inducing=0)
    with pytest.raises(ValueError):
        variational_gp_classifier(X, Y, lengthscale=0.0)
