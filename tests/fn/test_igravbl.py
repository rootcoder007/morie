"""Tests for morie.fn.igravbl: expected values recomputed from the defining equations."""

from morie.fn.igravbl import igravbl

F = [12.0, 0.0, 30.0, 2.0, 85.0, 4.0, 19.0, 1.0, 7.0, 40.0, 3.0, 16.0]
MO = [5.0, 5, 9, 9, 20, 20, 7, 7, 12, 12, 3, 3]
MD = [9.0, 20, 5, 20, 5, 9, 20, 9, 7, 3, 12, 5]
D = [1.0, 3.0, 1.0, 2.0, 3.0, 2.0, 2.5, 1.5, 2.0, 1.0, 3.0, 1.2]


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


def _ols(X, y):
    p = len(X[0])
    return _solve(
        [[sum(r[a] * r[b] for r in X) for b in range(p)] for a in range(p)],
        [sum(r[a] * t for r, t in zip(X, y)) for a in range(p)],
    )


def test_margins_and_odds_ratios():
    seed = [[1.0, 2.0, 0.5], [3.0, 4.0, 1.0], [0.2, 1.5, 2.0]]
    rows, cols = [10.0, 20.0, 5.0], [12.0, 15.0, 8.0]
    r = igravbl(seed, rows, cols)
    T = r.extra["balanced"]
    for i in range(3):
        assert abs(sum(T[i]) - rows[i]) < 1e-10
        assert abs(sum(T[k][i] for k in range(3)) - cols[i]) < 1e-10
    # IPF preserves every cross-product ratio of the seed
    for i, j, k, m in ((0, 1, 0, 1), (0, 2, 1, 2), (1, 2, 0, 2)):
        want = seed[i][k] * seed[j][m] / (seed[i][m] * seed[j][k])
        got = T[i][k] * T[j][m] / (T[i][m] * T[j][k])
        assert abs(got - want) < 1e-10 * want


def test_unequal_totals_rejected():
    import pytest

    with pytest.raises(ValueError):
        igravbl([[1, 1], [1, 1]], [1, 2], [1, 1])
