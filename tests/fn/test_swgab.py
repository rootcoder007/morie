"""Tests for morie.fn.swgab: recompute the Gabriel condition."""

import math

N = 12
C = [[math.cos(1.7 * i) * (1 + i / 6.0), math.sin(2.3 * i) + 0.1 * i] for i in range(N)]


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


from morie.fn.swgab import swgab  # noqa: E402


def test_gabriel():
    r = swgab(C)
    for i in range(N):
        want = [
            j
            for j in range(N)
            if j != i
            and not any(
                _d(C[i], C[k]) ** 2 + _d(C[j], C[k]) ** 2 < _d(C[i], C[j]) ** 2 for k in range(N) if k not in (i, j)
            )
        ]
        assert sorted(r.extra["neighbours"][i]) == want
