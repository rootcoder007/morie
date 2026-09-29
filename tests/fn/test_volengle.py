"""Tests for volengle.vol_engle_lagrange: the auxiliary regression recomputed."""

import math

from morie.fn import _array_core as np
from morie.fn.volengle import vol_engle_lagrange

R = [math.sin(1.3 * t) * (1 + 0.8 * math.sin(0.2 * t)) for t in range(80)]


def test_lm_statistic():
    q = 2
    m = sum(R) / 80
    e2 = [(v - m) ** 2 for v in R]
    Y = np.array(e2[q:])
    X = np.column_stack([np.ones(78), np.array(e2[1:79]), np.array(e2[0:78])])
    b = np.linalg.inv(X.T @ X) @ (X.T @ Y)
    res = Y - X @ b
    ym = float(Y.sum()) / 78
    r2 = 1 - float(res @ res) / float(((Y - ym) * (Y - ym)).sum())
    r = vol_engle_lagrange(R, q=2)
    assert abs(r["statistic"] - 78 * r2) < 1e-9
    assert abs(r["p_value"] - math.exp(-78 * r2 / 2)) < 1e-10
