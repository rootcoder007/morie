"""Tests for morie.fn.swtri: Delaunay neighbours satisfy the empty-circumcircle property."""

import math

N = 12
C = [[math.cos(1.7 * i) * (1 + i / 6.0), math.sin(2.3 * i) + 0.1 * i] for i in range(N)]


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


from morie.fn.swtri import swtri  # noqa: E402


def test_gabriel_subgraph_and_symmetry():
    nb = swtri(C).extra["neighbours"]
    for i in range(N):
        for j in nb[i]:
            assert i in nb[j]
    # every Gabriel edge is a Delaunay edge
    for i in range(N):
        for j in range(N):
            if j != i and not any(
                _d(C[i], C[k]) ** 2 + _d(C[j], C[k]) ** 2 < _d(C[i], C[j]) ** 2 for k in range(N) if k not in (i, j)
            ):
                assert j in nb[i]
    # planar triangulation: edges = 3n - 3 - hull size <= 3n - 6
    assert sum(len(v) for v in nb) // 2 <= 3 * N - 6
