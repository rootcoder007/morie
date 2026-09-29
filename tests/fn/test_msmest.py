"""Tests for morie.fn.msmest: stabilised weights and the weighted fit recomputed."""

import math

from morie.fn import _array_core as np
from morie.fn.msmest import marginal_structural_model

N = 40
L = [[math.sin(i), math.cos(0.7 * i)] for i in range(N)]
A = [
    [1.0 if L[i][0] + 1.5 * math.sin(3.3 * i + 1) > 0 else 0.0, 1.0 if L[i][1] + 1.5 * math.cos(2.1 * i) > 0 else 0.0]
    for i in range(N)
]
Y = [1.0 + 0.5 * (A[i][0] + A[i][1]) + L[i][0] + 0.2 * math.sin(5 * i) for i in range(N)]


def _logit(X, y):
    D = np.array([[1.0] + list(r) for r in X])
    b = np.zeros(D.shape[1])
    for _ in range(60):
        p = [1 / (1 + math.exp(-float(v))) for v in (D @ b).tolist()]
        W = np.diag(np.array([q * (1 - q) for q in p]))
        b = b + np.linalg.inv(D.T @ W @ D) @ (D.T @ (np.array(y) - np.array(p)))
    return [1 / (1 + math.exp(-float(v))) for v in (D @ b).tolist()]


def test_weights_and_effect():
    a0 = [r[0] for r in A]
    a1 = [r[1] for r in A]
    m = sum(a0) / N
    d0 = _logit([[L[i][0]] for i in range(N)], a0)
    n1 = _logit([[a0[i]] for i in range(N)], a1)
    d1 = _logit([[a0[i], L[i][0], L[i][1]] for i in range(N)], a1)
    sw = []
    for i in range(N):
        w = (m if a0[i] else 1 - m) / (d0[i] if a0[i] else 1 - d0[i])
        w *= (n1[i] if a1[i] else 1 - n1[i]) / (d1[i] if a1[i] else 1 - d1[i])
        sw.append(w)
    r = marginal_structural_model(Y, A, L)
    assert max(abs(a - b) for a, b in zip(r["weights"], sw)) < 1e-8
    D = np.column_stack([np.ones(N), np.array([a0[i] + a1[i] for i in range(N)])])
    Wd = np.diag(np.array(sw))
    b = np.linalg.inv(D.T @ Wd @ D) @ (D.T @ Wd @ np.array(Y))
    assert abs(r["estimate"] - float(b[1])) < 1e-8
    assert abs(r["intercept"] - float(b[0])) < 1e-8
