"""Tests for morie.fn.prnkpg: PageRank recomputed as the solution of its linear system."""

from morie.fn.prnkpg import pagerank


def _solve(A, b):
    n = len(A)
    M = [list(r) + [v] for r, v in zip(A, b)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [a - f * q for a, q in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def test_fixed_point_of_the_google_matrix():
    # x = (1 - d)/n + d P' x with P row-stochastic (no dangling nodes) and sum(x) = 1
    G = [[0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]]
    n, d = 4, 0.85
    P = [[v / sum(r) for v in r] for r in G]
    A = [[(1.0 if i == j else 0.0) - d * P[j][i] for j in range(n)] for i in range(n)]
    x = _solve(A, [(1 - d) / n] * n)
    r = pagerank(G)
    assert max(abs(a - b) for a, b in zip(r["pr"], x)) < 1e-12


def test_dangling_mass_redistributed():
    G = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
    pr = pagerank(G, n_iter=200)["pr"]
    assert abs(sum(pr) - 1.0) < 1e-12
