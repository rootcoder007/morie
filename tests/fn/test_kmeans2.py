"""Tests for morie.fn.kmeans2: a Lloyd fixed point with the recomputed inertia."""

import math

from morie.fn.kmeans2 import kmeans2

P = [[math.cos(k) + 4 * (k % 3), math.sin(k) + 3 * (k % 3 == 1)] for k in range(24)]


def test_fixed_point():
    r = kmeans2(P, n_clusters=3, n_init=5, random_state=7)
    lab, C = r["labels"], r["centroids"]
    for i, p in enumerate(P):
        d = [sum((a - b) ** 2 for a, b in zip(p, c)) for c in C]
        assert d[lab[i]] == min(d)
    for j in range(3):
        mem = [p for p, lb in zip(P, lab) if lb == j]
        assert all(abs(C[j][t] - sum(m[t] for m in mem) / len(mem)) < 1e-12 for t in range(2))
    sse = sum(sum((a - b) ** 2 for a, b in zip(p, C[lab[i]])) for i, p in enumerate(P))
    assert abs(r["inertia"] - sse) < 1e-12
