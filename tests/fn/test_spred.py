"""Tests for spred.shrinkage_predictor_level2 (BLUP of cluster means)."""

import pytest

from morie.fn.spred import shrinkage_predictor_level2


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


def _data():
    y = [1.2, 0.8, 1.5, 0.9, 1.1, 2.3, 2.0, 1.7, 2.6, 2.2, 1.9, 2.4, 2.1, -0.4, 0.3, 0.1]
    cl = [0] * 5 + [1] * 8 + [2] * 3
    return y, cl


def test_spred_basic():
    """Henderson's mixed model equations for y = mu + u_j + e with
    G = s2u I, R = s2e I give mu-hat and u-hat; the predicted cluster
    means mu-hat + u-hat_j are the shrunk values, and mu-hat is the
    reported grand mean (the GLS mean, not ybar.. -- the clusters are
    unbalanced)."""
    y, cl = _data()
    s2u, s2e = 0.5, 1.0
    q = 3
    # unknowns (mu, u0, u1, u2); C'C + diag(0, s2e/s2u, ...)
    rows = [[1.0] + [1.0 if c == j else 0.0 for j in range(q)] for c in cl]
    A = [[sum(r[a] * r[b] for r in rows) + (s2e / s2u if a == b and a > 0 else 0.0)
          for b in range(q + 1)] for a in range(q + 1)]
    sol = _solve(A, [sum(r[a] * t for r, t in zip(rows, y)) for a in range(q + 1)])
    r = shrinkage_predictor_level2(y, cl, s2u, s2e)
    assert r["grand_mean"] == pytest.approx(sol[0], abs=1e-12)
    assert r["shrunk"] == pytest.approx([sol[0] + u for u in sol[1:]], abs=1e-12)
    assert r["grand_mean"] != pytest.approx(sum(y) / len(y), abs=1e-3)
    sizes = [5.0, 8.0, 3.0]
    assert r["lambda"] == pytest.approx([s2e / (s2e + n * s2u) for n in sizes], abs=1e-15)
    assert r["sizes"] == sizes


def test_spred_edge():
    """s2u = 0 shrinks every cluster to ybar..; balanced clusters make
    the GLS mean equal ybar..; bad inputs raise."""
    y, cl = _data()
    r = shrinkage_predictor_level2(y, cl, 0.0, 1.0)
    assert r["shrunk"] == pytest.approx([sum(y) / len(y)] * 3, abs=1e-12)
    yb = [1.0, 2.0, 3.0, 5.0, 6.0, 10.0]
    rb = shrinkage_predictor_level2(yb, [0, 0, 1, 1, 2, 2], 2.0, 1.0)
    assert rb["grand_mean"] == pytest.approx(sum(yb) / 6, abs=1e-12)
    with pytest.raises(ValueError):
        shrinkage_predictor_level2(y, [0.5] * len(y), 1.0, 1.0)
    with pytest.raises(ValueError):
        shrinkage_predictor_level2(y, cl, -1.0, 1.0)
    with pytest.raises(ValueError):
        shrinkage_predictor_level2(y, [0] * len(y), 1.0, 1.0)
