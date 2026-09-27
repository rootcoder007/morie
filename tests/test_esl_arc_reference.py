"""Archetypal analysis (ESL 14.75-14.77): simplex constraints, monotone criterion, KKT, corners, R parity."""

import math

from morie.fn.eslarc import _simplex_ls, esl_archetypes


def data():
    return [[math.sin(1.7 * i) * 2 + math.cos(i), math.cos(2.3 * i) + 0.5 * math.sin(0.3 * i * i)] for i in range(50)]


def test_constraints_monotone_criterion_and_parity():
    X = data()
    r = esl_archetypes(X, 3)
    W, B, H = r["W"], r["B"], r["archetypes"]
    for row in W + B:
        assert min(row) >= 0 and abs(sum(row) - 1) < 1e-12
    for k in range(3):
        for c in range(2):
            assert abs(H[k][c] - sum(B[k][i] * X[i][c] for i in range(50))) < 1e-12  # H = BX (14.76)
    J = sum((X[i][c] - sum(W[i][k] * H[k][c] for k in range(3))) ** 2 for i in range(50) for c in range(2))
    assert abs(J - r["rss"]) < 1e-12  # (14.77)
    p = r["rss_path"]
    assert r["converged"] and all(b <= a + 1e-12 for a, b in zip(p, p[1:]))
    # the archetypes package (penalised NNLS, best of 20 seeds) reaches 7.608067 here
    assert r["rss"] < 7.608
    assert abs(r["rss"] - 5.1809409499909647) < 1e-9  # R arm


def test_simplex_least_squares_kkt():
    X = data()
    H = esl_archetypes(X, 3)["archetypes"]
    for x in (X[5], [3.0, 3.0], [0.1, -0.2]):
        w = _simplex_ls(H, x)
        g = [sum(H[k][c] * (sum(w[j] * H[j][c] for j in range(3)) - x[c]) for c in range(2)) for k in range(3)]
        supp = [k for k in range(3) if w[k] > 0]
        nu = -g[supp[0]]
        assert all(abs(g[k] + nu) < 1e-10 for k in supp)  # equal gradients on the support
        assert all(g[k] + nu > -1e-10 for k in range(3))  # non-negative multipliers off it


def test_corners_and_single_archetype():
    X = [[0, 0], [4, 0], [4, 3], [0, 3], [1, 1], [2, 2], [3, 1]]
    r = esl_archetypes(X, 4)
    assert sorted(map(tuple, r["archetypes"])) == [(0, 0), (0, 3), (4, 0), (4, 3)] and r["rss"] < 1e-24
    D = data()
    h = esl_archetypes(D, 1)["archetypes"][0]
    for c in range(2):
        assert abs(h[c] - sum(x[c] for x in D) / 50) < 1e-10  # one archetype = the centroid
