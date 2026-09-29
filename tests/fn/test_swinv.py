"""Tests for morie.fn.swinv: recompute d^-power."""

import math

N = 12
C = [[math.cos(1.7 * i) * (1 + i / 6.0), math.sin(2.3 * i) + 0.1 * i] for i in range(N)]


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


from morie.fn.swinv import swinv  # noqa: E402


def test_inverse_distance():
    W = swinv(C, power=1.5).extra["W"].tolist()
    for i in range(N):
        for j in range(N):
            ref = 0.0 if i == j else _d(C[i], C[j]) ** -1.5
            assert abs(W[i][j] - ref) < 1e-12 * max(1.0, ref)
    Wb = swinv(C, power=2.0, d=1.0).extra["W"].tolist()
    assert all(Wb[i][j] == 0.0 for i in range(N) for j in range(N) if _d(C[i], C[j]) > 1.0)
