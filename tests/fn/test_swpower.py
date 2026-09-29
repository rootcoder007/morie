"""Tests for morie.fn.swpower."""

from morie.fn.swpower import swpower

W = [[0, 0.5, 0.5, 0], [1 / 3, 0, 1 / 3, 1 / 3], [0.5, 0.5, 0, 0], [0, 1, 0, 0]]


def _mm(A, B):
    return [[sum(A[i][m] * B[m][j] for m in range(4)) for j in range(4)] for i in range(4)]


def test_powers():
    ref = _mm(_mm(W, W), W)
    P = swpower(W, 3)
    assert max(abs(P[i][j] - ref[i][j]) for i in range(4) for j in range(4)) < 1e-15
    assert swpower(W, 0) == [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    # row-stochastic W stays row-stochastic
    assert all(abs(sum(r) - 1.0) < 1e-14 for r in P)
