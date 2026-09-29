"""Tests for morie.fn.sfredu: recompute from the Moran eigenvectors."""

import math

from morie.fn import _array_core as np
from morie.fn.sfilter import moran_eigenvectors
from morie.fn.sfredu import sfredu

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


def test_before_after():
    r = sfredu(Y, W)
    assert abs(r["moran_before"] - _moran(Y)) < 1e-12
    E = moran_eigenvectors(W)["vectors"]
    _, e = _ols_resid([[E[i][k] for i in range(N)] for k in r["selected"]], Y)
    assert abs(r["moran_after"] - _moran(e)) < 1e-10
    assert abs(r["reduction"] - (r["moran_before"] - r["moran_after"])) < 1e-15
