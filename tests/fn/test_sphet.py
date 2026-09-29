"""Tests for morie.fn.sphet: Koenker n R^2 recomputed."""

import math

from morie.fn import _array_core as np
from morie.fn.sphet import spatial_heterogeneity

N = 12
W = [[0.5 if abs(i - j) == 1 else 0.0 for j in range(N)] for i in range(N)]
W[0][1] = W[N - 1][N - 2] = 1.0
E = [math.sin(1.7 * i) * (1 + i / 4.0) for i in range(N)]


def _nr2(Z):
    e2 = np.array([v * v for v in E])
    D = np.column_stack([np.ones(N)] + Z)
    fit = D @ (np.linalg.inv(D.T @ D) @ (D.T @ e2))
    r = e2 - fit
    c = e2 - float(e2.sum()) / N
    return N * (1 - float(r @ r) / float(c @ c))


def test_default_z_is_lagged_squares():
    e2 = [v * v for v in E]
    z = np.array(W) @ np.array(e2)
    ref = _nr2([z])
    r = spatial_heterogeneity(E, W)
    assert abs(r.statistic - ref) < 1e-10
    assert abs(r.p_value - math.erfc(math.sqrt(ref / 2))) < 1e-10


def test_user_z():
    xs = np.array([float(i) for i in range(N)])
    ref = _nr2([xs])
    assert abs(spatial_heterogeneity(E, W, Z=[[float(i)] for i in range(N)]).statistic - ref) < 1e-10
