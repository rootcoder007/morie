"""Tests for morie.fn.sfmemb: recompute from the Moran eigenvectors."""

import math

from morie.fn import _array_core as np
from morie.fn.sfilter import moran_eigenvectors
from morie.fn.sfmemb import sfmemb

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


def _p_t10(t):
    """Two-sided p of a t with 10 df: 1 - A(t|10), A = sin(q)(1 + c/2 + 3c^2/8 + 5c^3/16 + 35c^4/128), c = cos(q)^2."""
    q = math.atan(abs(t) / math.sqrt(10))
    c = math.cos(q) ** 2
    return 1 - math.sin(q) * (1 + c / 2 + 3 * c * c / 8 + 5 * c**3 / 16 + 35 * c**4 / 128)


def test_bonferroni_selection():
    r = sfmemb(Y, W, alpha=0.1)
    me = moran_eigenvectors(W)
    m = len(me["positive"])
    yc = [v - sum(Y) / N for v in Y]
    ss = math.sqrt(sum(v * v for v in yc))
    for idx, k in enumerate(me["positive"]):
        rr = sum(me["vectors"][i][k] * yc[i] for i in range(N)) / ss
        t = rr * math.sqrt((N - 2) / (1 - rr * rr))
        assert abs(r["t"][idx] - t) < 1e-10
        assert abs(r["p_values"][idx] - _p_t10(t)) < 1e-10
        assert (k in r["selected"]) == (_p_t10(t) < 0.1 / m)
