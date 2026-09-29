"""Tests for morie.fn.gwrres: recompute the local weighted least squares in matrix form."""

import math

from morie.fn import _array_core as np
from morie.fn.gwrres import gwrres

N = 16
P = [(float(i % 4), float(i // 4)) for i in range(N)]
X = [[(0.3 * i) % 1.7] for i in range(N)]
Y = [1.0 + 2.0 * X[i][0] + 0.1 * P[i][0] + 0.05 * (i % 3) + 0.2 * math.sin(i) for i in range(N)]
BW = 2.5


def _w(i):
    """Gaussian kernel weights of every point at location i."""
    return [math.exp(-(math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]) ** 2) / (2 * BW * BW)) for j in range(N)]


def _local(i, w=None):
    """(beta, C) of the local WLS at location i, in matrix form."""
    Xa = np.column_stack([np.ones(N), np.array([r[0] for r in X])])
    Wd = np.diag(np.array(_w(i) if w is None else w))
    C = np.linalg.inv(Xa.T @ Wd @ Xa) @ Xa.T @ Wd
    return C @ np.array(Y), C, Xa


def test_residuals():
    e = gwrres(Y, X, P, BW, kernel="gaussian")
    for i in (0, 7):
        b, _, Xa = _local(i)
        assert abs(e[i] - (Y[i] - float(Xa[i] @ b))) < 1e-10
