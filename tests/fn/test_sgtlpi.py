"""Tests for sgtlpi.sgt_laplacian_pseudoinverse."""

from morie.fn.sgtlpi import sgt_laplacian_pseudoinverse


def _cycle(n):
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        A[i][(i + 1) % n] = A[(i + 1) % n][i] = 1.0
    return A


def _laplacian(A):
    n = len(A)
    return [[(sum(A[i]) if i == j else 0.0) - A[i][j] for j in range(n)] for i in range(n)]


def _matmul(P, Q):
    n, m, k = len(P), len(Q[0]), len(Q)
    return [[sum(P[i][t] * Q[t][j] for t in range(k)) for j in range(m)] for i in range(n)]


def test_sgtlpi_basic():
    """L+ satisfies the Moore-Penrose identity L L+ L = L and has zero row sums."""
    A = _cycle(5)
    r = sgt_laplacian_pseudoinverse(A)
    assert r["n"] == 5
    assert r["rank"] == 4
    L = _laplacian(A)
    Lp = [list(map(float, row)) for row in r["Lplus"]]
    back = _matmul(_matmul(L, Lp), L)
    for i in range(5):
        assert abs(sum(Lp[i])) < 1e-9
        for j in range(5):
            assert abs(back[i][j] - L[i][j]) < 1e-9


def test_sgtlpi_eigenvalues():
    """The cycle C_4 Laplacian spectrum is 0, 2, 2, 4."""
    r = sgt_laplacian_pseudoinverse(_cycle(4))
    ev = sorted(float(v) for v in r["eigenvalues"])
    expected = [0.0, 2.0, 2.0, 4.0]
    for got, want in zip(ev, expected):
        assert abs(got - want) < 1e-9


def test_sgtlpi_edge():
    """L+ is symmetric, so it is unchanged by transposition."""
    r = sgt_laplacian_pseudoinverse(_cycle(6))
    Lp = [list(map(float, row)) for row in r["Lplus"]]
    for i in range(6):
        for j in range(6):
            assert abs(Lp[i][j] - Lp[j][i]) < 1e-9
