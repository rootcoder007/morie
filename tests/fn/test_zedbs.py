"""Tests for morie.fn.zedbs: core points and cluster labels recomputed."""

import math

from morie.fn.zedbs import spatial_dbscan

P = [[math.cos(k) + 4 * (k % 3), math.sin(k) + 3 * (k % 3 == 1)] for k in range(24)] + [[20.0, 20.0]]


def test_labels():
    r = spatial_dbscan(P, eps=1.8, min_pts=4)
    lab = r.extra["cluster"]
    n = len(P)
    nb = [[j for j in range(n) if math.dist(P[i], P[j]) <= 1.8] for i in range(n)]
    core = [len(v) >= 4 for v in nb]
    assert lab[-1] == 0 and r.extra["n_noise"] >= 1
    for i in range(n):
        if core[i]:
            assert all(lab[j] == lab[i] for j in nb[i] if core[j])
    assert r.statistic == max(lab)
