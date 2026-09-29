"""mapeq: map equation identities and planted communities."""

import math

import pytest

from morie.fn.mapeq import infomap_partition, map_equation


def _cliques(k, size):
    n = k * size
    A = [[0.0] * n for _ in range(n)]
    for c in range(k):
        for i in range(c * size, (c + 1) * size):
            for j in range(c * size, (c + 1) * size):
                if i != j:
                    A[i][j] = 1.0
    for c in range(k):
        a, b = c * size, ((c + 1) % k) * size + 1
        A[a][b] = A[b][a] = 1.0
    return A


def test_one_module_is_the_visit_entropy():
    A = _cliques(3, 4)
    s = [sum(r) for r in A]
    p = [v / sum(s) for v in s]
    H = -sum(v * math.log2(v) for v in p)
    assert map_equation(A, [0] * 12) == pytest.approx(H, rel=1e-14)


def test_planted_cliques_are_found():
    A = _cliques(4, 5)
    r = infomap_partition(A)
    assert r.membership == [i // 5 for i in range(20)]
    assert r.codelength == pytest.approx(map_equation(A, r.membership), rel=1e-14)
    assert r.codelength < map_equation(A, [0] * 20)
    t = infomap_partition(A, "igraph")
    assert t.membership == r.membership
    with pytest.raises(ValueError):
        map_equation(A, [0] * 20, "directed")
