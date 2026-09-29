"""Tests for morie.fn.swdist: recompute the distance band."""

import math

N = 12
C = [[math.cos(1.7 * i) * (1 + i / 6.0), math.sin(2.3 * i) + 0.1 * i] for i in range(N)]


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


from morie.fn.swdist import swdist  # noqa: E402


def test_band():
    r = swdist(C, d=1.1, d_min=0.2)
    ref = [[1.0 if i != j and 0.2 < _d(C[i], C[j]) <= 1.1 else 0.0 for j in range(N)] for i in range(N)]
    assert [list(map(float, row)) for row in r.extra["W"].tolist()] == ref
    assert abs(r.statistic - sum(map(sum, ref)) / N) < 1e-15
