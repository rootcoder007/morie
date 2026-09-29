"""Tests for morie.fn.sfloc: recompute from the Moran eigenvectors."""

import math

from morie.fn import _array_core as np
from morie.fn.sfilter import moran_eigenvectors
from morie.fn.sfloc import sfloc

N = 12
W = [[1.0 if abs(i - j) == 1 or abs(i - j) == 4 else 0.0 for j in range(N)] for i in range(N)]
Y = [1.0 + math.sin(0.6 * i) + 0.3 * math.cos(2.1 * i) for i in range(N)]


def _moran(e):
    m = sum(e) / N
    d = [v - m for v in e]
    s0 = sum(map(sum, W))
    return N / s0 * sum(d[i] * W[i][j] * d[j] for i in range(N) for j in range(N)) / sum(v * v for v in d)


def _ols_resid(cols, y):
    Xa = np.column_stack([np.ones(N)] + [np.array(c) for c in cols])
    b = np.linalg.inv(Xa.T @ Xa) @ (Xa.T @ np.array(y))
    return b, (np.array(y) - Xa @ b).tolist()


def test_filter_is_E_gamma():
    X = [[0.1 * k + math.cos(k)] for k in range(N)]
    r = sfloc(Y, X, W, i=5, criterion="aic")
    E = moran_eigenvectors(W)["vectors"]
    cols = [[x[0] for x in X]] + [[E[i][k] for i in range(N)] for k in r["selected"]]
    b, _ = _ols_resid(cols, Y)
    g = b.tolist()[2:]
    f5 = sum(E[5][k] * gk for k, gk in zip(r["selected"], g))
    assert abs(r["value"] - f5) < 1e-10
    assert abs(r["filter"][5] - r["value"]) == 0.0
