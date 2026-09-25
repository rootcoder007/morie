"""Tests for tagRC.folkrank (Hotho, Jaschke, Schmitz & Stumme 2006)."""

import pytest

from morie.fn.tagRC import folkrank, preference_vector, tripartite_graph


TRIPLES = [("u1", "python", "r1"), ("u1", "stats", "r2"), ("u2", "python", "r1"),
           ("u2", "python", "r3"), ("u3", "cooking", "r4"), ("u3", "python", "r4"),
           ("u4", "stats", "r2"), ("u4", "cooking", "r5"), ("u5", "python", "r3")]


def _solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def test_tagRC_basic():
    """w1 solves w = d A w + (1 - d) p exactly (A the degree-normalised
    undirected adjacency, eq. 1 with alpha = 0); w0 is the beta = 1 fixed
    point, the degree distribution (sec. 4.1, step 2); FolkRank is
    w1 - w0."""
    g = tripartite_graph(TRIPLES)
    N, adj = g["nodes"], g["adjacency"]
    deg = {u: sum(adj[u].values()) for u in N}
    p = preference_vector(N, ["t:cooking"], 0.9)["p"]
    d = 0.7
    A = [[(1.0 if i == j else 0.0) - d * adj.get(v, {}).get(u, 0.0) / deg[v]
          for j, v in enumerate(N)] for i, u in enumerate(N)]
    w1 = _solve(A, [(1 - d) * p[u] for u in N])
    w0 = [deg[u] / sum(deg.values()) for u in N]
    r = folkrank(TRIPLES, ["t:cooking"], d=d, weight=0.9, iters=2000)
    for i, u in enumerate(N):
        assert r["with_preference"][u] == pytest.approx(w1[i], abs=1e-11)
        assert r["without_preference"][u] == pytest.approx(w0[i], abs=1e-15)
        assert r["difference"][u] == pytest.approx(w1[i] - w0[i], abs=1e-11)
    assert r["ranking"][0] == "t:cooking"
    # the globally dominant tag leads the undifferenced ranking only
    assert r["baseline_ranking"][0] == "t:python"


def test_tagRC_edge():
    """Each triple adds three undirected edges; focus nodes absent from
    the graph and weights outside (0, 1) raise."""
    g = tripartite_graph([("a", "t", "r")])
    assert g["adjacency"]["u:a"] == {"t:t": 1.0, "r:r": 1.0}
    with pytest.raises(ValueError):
        folkrank(TRIPLES, ["t:nothing"])
    with pytest.raises(ValueError):
        preference_vector(["x"], ["x"], weight=1.0)
