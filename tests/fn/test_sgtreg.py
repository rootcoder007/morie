"""Tests for sgtreg.sgt_resistance_distance_matrix."""

from morie.fn.sgtreg import sgt_resistance_distance_matrix


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


def test_sgtreg_basic():
    """On a path each edge is a unit resistor in series: R_ij = |i - j|."""
    n = 5
    r = sgt_resistance_distance_matrix(_path(n))
    assert r["n"] == n
    assert r["rank"] == n - 1
    R = [list(map(float, row)) for row in r["R"]]
    for i in range(n):
        for j in range(n):
            assert abs(R[i][j] - abs(i - j)) < 1e-9


def test_sgtreg_cycle_and_complete():
    """R_ij = d(n-d)/n on C_n and 2/n on K_n."""
    n = 6
    R = [list(map(float, row)) for row in sgt_resistance_distance_matrix(_cycle(n))["R"]]
    for i in range(n):
        for j in range(n):
            d = min((i - j) % n, (j - i) % n)
            assert abs(R[i][j] - d * (n - d) / n) < 1e-9
    Rk = [list(map(float, row)) for row in sgt_resistance_distance_matrix(_complete(n))["R"]]
    for i in range(n):
        for j in range(n):
            assert abs(Rk[i][j] - (0.0 if i == j else 2.0 / n)) < 1e-9


def test_sgtreg_edge():
    """Adding a parallel route lowers the resistance below the path length."""
    path = sgt_resistance_distance_matrix(_path(4))["R"]
    cyc = sgt_resistance_distance_matrix(_cycle(4))["R"]
    assert abs(float(path[0][3]) - 3.0) < 1e-9
    assert abs(float(cyc[0][3]) - 0.75) < 1e-9
