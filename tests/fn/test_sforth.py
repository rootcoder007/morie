"""Tests for morie.fn.sforth: recompute from the Moran eigenvectors."""

import math

from morie.fn import _array_core as np
from morie.fn.sfilter import moran_eigenvectors
from morie.fn.sforth import sforth

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


def test_mems_are_orthonormal():
    me = moran_eigenvectors(W)
    E = [[me["vectors"][i][k] for k in me["positive"]] for i in range(N)]
    r = sforth(E)
    assert r.statistic < 1e-12 and r.extra["orthonormal"]


def test_detects_non_orthogonal():
    E = [[1.0, 1.0], [0.0, 1.0]]
    r = sforth(E)
    assert abs(r.statistic - 1.0) < 1e-15
    assert abs(r.extra["max_constant_projection"] - 2 / math.sqrt(2)) < 1e-15
