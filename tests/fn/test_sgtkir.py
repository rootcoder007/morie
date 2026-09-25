"""Tests for sgtkir.sgt_kirchhoff_index."""

from morie.fn.sgtkir import sgt_kirchhoff_index


def _path(n):
    A = [[0.0] * n for _ in range(n)]
    for i in range(n - 1):
        A[i][i + 1] = A[i + 1][i] = 1.0
    return A


def _cycle(n):
    A = _path(n)
    A[0][n - 1] = A[n - 1][0] = 1.0
    return A


def _complete(n):
    return [[0.0 if i == j else 1.0 for j in range(n)] for i in range(n)]


def test_sgtkir_basic():
    """Complete graph: Kf(K_n) = n - 1 (Klein & Randic 1993)."""
    for n in (3, 4, 6):
        r = sgt_kirchhoff_index(_complete(n))
        assert r["n"] == n
        assert r["rank"] == n - 1
        assert abs(r["Kf"] - (n - 1)) < 1e-9
        # the paper's identity: the double-sum form equals the spectral form
        assert abs(r["Kf"] - r["Kf_spectral"]) < 1e-9


def test_sgtkir_path_and_cycle():
    """Kf(P_n) = n(n^2-1)/6 and Kf(C_n) = n(n^2-1)/12."""
    for n in (4, 5, 6):
        rp = sgt_kirchhoff_index(_path(n))
        assert abs(rp["Kf"] - n * (n * n - 1) / 6.0) < 1e-9
        rc = sgt_kirchhoff_index(_cycle(n))
        assert abs(rc["Kf"] - n * (n * n - 1) / 12.0) < 1e-9


def test_sgtkir_edge():
    """Disconnected graph: the rank drops by one per extra component."""
    A = [[0.0, 1.0, 0.0, 0.0],
         [1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0],
         [0.0, 0.0, 1.0, 0.0]]
    r = sgt_kirchhoff_index(A)
    assert r["n"] == 4
    assert r["rank"] == 2
