"""Tests for morie.fn.sacjac: every expected value is recomputed from the formula."""

import math

from morie.fn.sacjac import sacjac


def _inv(A):
    n = len(A)
    M = [list(map(float, r)) + [1.0 if i == j else 0.0 for j in range(n)] for i, r in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        d = M[c][c]
        M[c] = [v / d for v in M[c]]
        for r in range(n):
            if r != c:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [r[n:] for r in M]


def _mm(A, B):
    return [[sum(a * b for a, b in zip(r, c)) for c in zip(*B)] for r in A]


def _mv(A, v):
    return [sum(a * b for a, b in zip(r, v)) for r in A]


def _t(A):
    return [list(c) for c in zip(*A)]


def _imw(W, a):
    return [[(1.0 if i == j else 0.0) - a * W[i][j] for j in range(len(W))] for i in range(len(W))]


def _det(A):
    # Laplace expansion (small matrices only)
    if len(A) == 1:
        return A[0][0]
    return sum((-1) ** j * A[0][j] * _det([r[:j] + r[j + 1 :] for r in A[1:]]) for j in range(len(A)))


N = 8
_B = [[1.0 if abs(i - j) == 1 or {i, j} == {0, 7} or {i, j} == {2, 5} else 0.0 for j in range(N)] for i in range(N)]
W = [[v / sum(r) for v in r] for r in _B]
X = [[1.0, ((i * 7) % 11) / 5] for i in range(N)]
Y = [1 + 2 * X[i][1] + ((i * 3) % 5 - 2) / 4 + 0.3 * sum(W[i][j] * X[j][1] for j in range(N)) for i in range(N)]
E = [((i * 4) % 7 - 3) / 5 for i in range(N)]
W3 = [[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]]


def test_sum_of_two_log_determinants():
    r = sacjac(W, 0.4, -0.3)
    a = math.log(abs(_det(_imw(W, 0.4))))
    b = math.log(abs(_det(_imw(W, -0.3))))
    assert abs(r.extra["logdet_rho"] - a) < 1e-11
    assert abs(r.extra["logdet_lambda"] - b) < 1e-11
    assert abs(r.statistic - (a + b)) < 1e-11


def test_two_by_two_closed_form():
    # det(I - aW) = 1 - a^2 for W = [[0, 1], [1, 0]]
    r = sacjac([[0, 1], [1, 0]], 0.5, 0.2)
    assert abs(r.statistic - (math.log(0.75) + math.log(0.96))) < 1e-14
