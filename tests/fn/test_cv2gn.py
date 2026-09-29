"""Tests for cv2gn.cv2_genomic: one fold's ridge fit and correlation recomputed."""

import math

from morie.fn import _array_core as np
from morie.fn.cv2gn import cv2_genomic

M = [[math.sin(1.1 * i + j) for j in range(4)] for i in range(10)]
Y = [
    sum(M[i % 10][j] * (j + 1) for j in range(4)) + (0.5 if i >= 10 else 0.0) + 0.1 * math.cos(3 * i) for i in range(20)
]
MK = [M[i % 10] for i in range(20)]
ENV = [i // 10 for i in range(20)]


def test_fold_recompute():
    r = cv2_genomic(Y, MK, ENV, n_folds=4, lam=0.3, seed=5)
    f = 2
    tr = [i for i in range(20) if r["folds"][i] != f]
    te = [i for i in range(20) if r["folds"][i] == f]
    D = [[1.0, float(ENV[i] == 1)] + MK[i] for i in range(20)]
    A = np.array([D[i] for i in tr])
    P = np.diag(np.array([0.0, 0.0] + [0.3] * 4))
    b = np.linalg.inv(A.T @ A + P) @ (A.T @ np.array([Y[i] for i in tr]))
    pred = [float(np.array(D[i]) @ b) for i in te]
    obs = [Y[i] for i in te]
    mp, mo = sum(pred) / len(pred), sum(obs) / len(obs)
    cor = sum((a - mp) * (c - mo) for a, c in zip(pred, obs)) / math.sqrt(
        sum((a - mp) ** 2 for a in pred) * sum((c - mo) ** 2 for c in obs)
    )
    assert abs(r["pa_folds"][f] - cor) < 1e-10
    assert sorted(r["folds"].count(k) for k in range(4)) == [5, 5, 5, 5]
