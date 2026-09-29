"""Tests for morie.fn.swkern: recompute the Gaussian and quartic kernels."""

import math

N = 12
C = [[math.cos(1.7 * i) * (1 + i / 6.0), math.sin(2.3 * i) + 0.1 * i] for i in range(N)]


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


from morie.fn.swkern import swkern  # noqa: E402


def test_kernels():
    G = swkern(C, bw=0.8).extra["W"].tolist()
    Q = swkern(C, bw=0.8, kernel="quartic").extra["W"].tolist()
    for i in range(N):
        for j in range(N):
            u = _d(C[i], C[j]) / 0.8
            g = 0.0 if i == j else math.exp(-0.5 * u * u) / math.sqrt(2 * math.pi)
            q = 0.0 if i == j or u >= 1 else 15.0 / 16.0 * (1 - u * u) ** 2
            assert abs(G[i][j] - g) < 1e-14
            assert abs(Q[i][j] - q) < 1e-14
