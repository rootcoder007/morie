"""Tests for morie.fn.swknn: recompute the k nearest neighbours."""

import math

N = 12
C = [[math.cos(1.7 * i) * (1 + i / 6.0), math.sin(2.3 * i) + 0.1 * i] for i in range(N)]


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


from morie.fn.swknn import swknn  # noqa: E402


def test_knn():
    r = swknn(C, k=3)
    for i in range(N):
        order = sorted((j for j in range(N) if j != i), key=lambda j: (_d(C[i], C[j]), j))
        assert sorted(r.extra["neighbours"][i]) == sorted(order[:3])
