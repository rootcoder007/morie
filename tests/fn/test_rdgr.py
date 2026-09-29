"""Tests for morie.fn.rdgr: the ridge normal equations."""

import math

from morie.fn import _array_core as np
from morie.fn.rdgr import rdgr

X = [[math.sin(k), math.cos(0.7 * k)] for k in range(15)]
Y = [1.0 + 2.0 * r[0] - r[1] + 0.1 * math.sin(3 * k) for k, r in enumerate(X)]


def test_centred_solution():
    Xa = np.array(X)
    xm = [sum(r[j] for r in X) / 15 for j in range(2)]
    ym = sum(Y) / 15
    Xc = Xa - np.array([xm] * 15)
    b = np.linalg.inv(Xc.T @ Xc + 0.7 * np.eye(2)) @ (Xc.T @ (np.array(Y) - ym))
    r = rdgr(X, Y, alpha=0.7)
    assert max(abs(float(b[j]) - r["coef"][j]) for j in range(2)) < 1e-12
    assert abs(r["intercept"] - (ym - xm[0] * float(b[0]) - xm[1] * float(b[1]))) < 1e-12
