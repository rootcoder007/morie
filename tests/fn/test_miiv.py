"""Tests for morie.fn.miiv: recompute the Anselin-Kelejian statistic."""

import math

from morie.fn import _array_core as np
from morie.fn.miiv import miiv

N = 8
W = [[0.5 if abs(i - j) == 1 else 0.0 for j in range(N)] for i in range(N)]
W[0][1] = W[N - 1][N - 2] = 1.0
Z = [[1.0, (i * 3 % 7) / 2.0, 0.2 * i + (i % 3)] for i in range(N)]
H = [[1.0, (i * 3 % 7) / 2.0, (i * i % 5) / 3.0, float(i % 2)] for i in range(N)]
E = [0.3, -0.2, 0.5, -0.9, 0.1, 0.4, -0.6, 0.35]


def test_ak_statistic():
    Wa, Za, Ha, e = np.array(W), np.array(Z), np.array(H), np.array(E)
    n = N
    s0 = float(Wa.sum())
    ete = float(e @ e)
    mi = n / s0 * float(e @ (Wa @ e)) / ete
    T = float(np.trace((Wa.T + Wa) @ Wa))
    V = np.linalg.inv(Za.T @ Ha @ np.linalg.inv(Ha.T @ Ha) @ Ha.T @ Za)
    g = (Wa.T @ e) @ Za
    phi2 = (T + 4.0 / (ete / n) * float(g @ (V @ g))) / ((s0 / n) ** 2 * n)
    ak = n * mi * mi / phi2
    r = miiv(E, W, Z, H)
    assert abs(r.statistic - ak) < 1e-12
    assert abs(r.extra["moran_i"] - mi) < 1e-13
    assert abs(r.p_value - math.erfc(math.sqrt(ak / 2))) < 1e-12
