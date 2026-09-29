"""Tests for morie.fn.swadapt: bandwidth = distance to the k-th neighbour."""

import math

N = 12
C = [[math.cos(1.7 * i) * (1 + i / 6.0), math.sin(2.3 * i) + 0.1 * i] for i in range(N)]


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


from morie.fn.swadapt import swadapt  # noqa: E402


def test_adaptive_bandwidth():
    W = swadapt(C, k=3, kernel="quartic").extra["W"].tolist()
    for i in range(N):
        h = sorted(_d(C[i], C[j]) for j in range(N) if j != i)[2]
        for j in range(N):
            u = _d(C[i], C[j]) / h
            q = 0.0 if i == j or u >= 1 else 15.0 / 16.0 * (1 - u * u) ** 2
            assert abs(W[i][j] - q) < 1e-14
