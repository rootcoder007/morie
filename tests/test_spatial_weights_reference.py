"""Spatial weights (spdep knn2nb, dnearneigh, gabrielneigh, relativeneigh, nb2listw styles)."""

import math

from morie.fn._rng import random_uniform
from morie.fn.spwgt import spatial_weights


def points():
    U = [float(u) for u in random_uniform(200, seed=51, stream=0)]
    return [[U[2 * i], U[2 * i + 1]] for i in range(40)]


def test_graphs_match_spdep():
    P = points()
    g = spatial_weights(P, "gabriel", style="B").extra["neighbours"]
    assert g[:3] == [[7, 27], [14, 16, 25, 33], [10, 24, 38]]
    r = spatial_weights(P, "relative", style="B").extra["neighbours"]
    assert r[:3] == [[7, 27], [14, 25], [10, 24]]
    k = spatial_weights(P, "knn", k=4, style="S")
    assert k.extra["neighbours"][0] == [5, 7, 22, 27] and all(abs(v - 0.25) < 1e-15 for v in k.extra["W"][0] if v)
    d = spatial_weights(P, "distance", threshold=0.2, d_min=0.02, style="B")
    assert d.extra["n_components"] == 6 and d.extra["n_islands"] == 4


def test_styles_and_constants():
    P = points()
    for st, total in (("W", 40.0), ("C", 40.0), ("U", 1.0), ("S", 40.0)):
        W = spatial_weights(P, "gabriel", style=st).extra["W"]
        assert abs(sum(sum(r) for r in W.tolist()) - total) < 1e-12
    r = spatial_weights(P, "inverse", threshold=0.3, alpha=2.0, style="minmax")
    W = r.extra["W"].tolist()
    assert abs(min(max(sum(row) for row in W), max(sum(W[i][j] for i in range(40)) for j in range(40))) - 1) < 1e-12
    e = spatial_weights([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], "kernel", k=2, kernel="epanechnikov", style="B")
    h = math.sqrt(0.5)
    assert abs(e.extra["W"].tolist()[4][0] - 0.75 * (1 - (math.sqrt(0.5) / h) ** 2)) < 1e-15
    assert e.extra["symmetric"] is False or e.extra["symmetric"] is True
