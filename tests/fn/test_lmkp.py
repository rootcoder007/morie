"""Tests for morie.fn.lmkp: spatial 2SLS residuals and the Anselin-Kelejian statistic recomputed."""

import math

from morie.fn import _array_core as np
from morie.fn.lmkp import lmkp

N = 14
W = [[0.0] * N for _ in range(N)]
for i in range(N):
    for d in (1, 2):
        W[i][(i + d) % N] = W[i][(i - d) % N] = 0.25
X = [[math.sin(i * 1.3) + i / 7.0, ((i * 5) % 7) / 3.0] for i in range(N)]
Y = [1.0 + 2.0 * X[i][0] - X[i][1] + ((i * 3) % 5 - 2) / 3.0 + 0.4 * math.cos(i) for i in range(N)]


def test_kp_statistic():
    Wa, ya = np.array(W), np.array(Y)
    Xa = np.column_stack([np.ones(N), np.array([r[0] for r in X]), np.array([r[1] for r in X])])
    WX = Wa @ Xa[:, 1:]
    H = np.column_stack([Xa, WX[:, 0], Wa @ WX[:, 0], WX[:, 1], Wa @ WX[:, 1]])
    Z = np.column_stack([Wa @ ya, Xa])
    P = H @ np.linalg.inv(H.T @ H) @ H.T
    Zh = P @ Z
    b = np.linalg.inv(Zh.T @ Zh) @ (Zh.T @ ya)
    e = ya - Z @ b
    s0 = float(Wa.sum())
    mi = N / s0 * float(e @ (Wa @ e)) / float(e @ e)
    T = float(np.trace((Wa.T + Wa) @ Wa))
    V = np.linalg.inv(Z.T @ P @ Z)
    g = (Wa.T @ e) @ Z
    phi2 = (T + 4.0 / (float(e @ e) / N) * float(g @ (V @ g))) / ((s0 / N) ** 2 * N)
    ak = N * mi * mi / phi2
    r = lmkp(Y, X, W)
    assert abs(r.statistic - ak) < 1e-9 * max(1.0, ak)
    assert abs(r.extra["rho"] - float(b[0])) < 1e-9
