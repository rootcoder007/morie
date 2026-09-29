"""Tests for morie.fn.swlagf: (I - rho W)^{-1} solves the reduced form."""

from morie.fn.swlagf import swlagf

W = [[0, 0.5, 0.5, 0], [1 / 3, 0, 1 / 3, 1 / 3], [0.5, 0.5, 0, 0], [0, 1, 0, 0]]


def test_inverse_and_series():
    rho = 0.4
    M = swlagf(W, rho)
    for i in range(4):
        for j in range(4):
            v = M[i][j] - rho * sum(M[i][m] * W[m][j] for m in range(4))
            assert abs(v - (1.0 if i == j else 0.0)) < 1e-14
    # Neumann series sum_p rho^p W^p
    P = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    S = [r[:] for r in P]
    for p in range(1, 80):
        P = [[sum(P[i][m] * W[m][j] for m in range(4)) for j in range(4)] for i in range(4)]
        S = [[S[i][j] + rho**p * P[i][j] for j in range(4)] for i in range(4)]
    assert max(abs(S[i][j] - M[i][j]) for i in range(4) for j in range(4)) < 1e-12
