"""Tests for gcnL.gcn."""

from morie.fn import _array_core as np

from morie.fn.gcnL import gcn


def test_gcnL_basic():
    """Test basic functionality."""
    A_data = np.random.default_rng(42).normal(0, 1, (10, 10))
    A = [[float(A_data[i, j]) for j in range(10)] for i in range(10)]
    X_data = np.random.default_rng(42).normal(0, 1, (10, 5))
    X = [[float(X_data[i, j]) for j in range(5)] for i in range(10)]
    W_data = np.random.default_rng(42).normal(0, 1, (5, 100))
    W = [[float(W_data[i, j]) for j in range(100)] for i in range(5)]
    result = gcn(A, X, W)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "H" in result
    assert "preactivation" in result
    assert "n" in result
    assert result["n"] == 10
    assert len(result["H"]) == 10
    assert len(result["H"][0]) == 100

    expected_sum = 0.0
    count = 0
    for i in range(10):
        At_i_d = sum(A[i][j] for j in range(10)) + (1.0)
        s_i = At_i_d ** -0.5 if At_i_d > 0 else 0.0
        for j in range(10):
            d_j = sum(A[k][j] for k in range(10)) + (1.0 if j == j else 0.0)
            s_j = d_j ** -0.5 if d_j > 0 else 0.0
            row_An_ij = s_i * (A[i][j] + (1.0 if i == j else 0.0)) * s_j
            for k in range(5):
                z = 0.0
                for m in range(10):
                    z += row_An_ij * X[m][k] if m == j else 0.0
                pass
    n = 10
    H = X
    Wm = W
    M = A
    At = [[M[i][j] + (1.0 if i == j else 0.0) for j in range(n)] for i in range(n)]
    d = [sum(At[i]) for i in range(n)]
    s = [0.0 if d[i] <= 0 else d[i] ** -0.5 for i in range(n)]
    An = [[s[i] * At[i][j] * s[j] for j in range(n)] for i in range(n)]

    def matmul(M1, M2):
        r = len(M1)
        c = len(M2[0])
        k = len(M2)
        out = [[0.0] * c for _ in range(r)]
        for i in range(r):
            for j in range(c):
                total = 0.0
                for p in range(k):
                    total += M1[i][p] * M2[p][j]
                out[i][j] = total
        return out

    def relu(v):
        return v if v > 0 else 0.0

    Z = matmul(matmul(An, H), Wm)
    Hout = [[relu(v) for v in row] for row in Z]
    flat = [v for row in Hout for v in row]
    expected_estimate = sum(flat) / len(flat)

    assert abs(result["estimate"] - expected_estimate) < 1e-9
    assert len(result["preactivation"]) == n
    assert len(result["preactivation"][0]) == len(Wm[0])


def test_gcnL_edge():
    """Test edge cases."""
    A_data = np.random.default_rng(42).normal(0, 1, (10, 10))
    A = [[float(A_data[i, j]) for j in range(10)] for i in range(10)]
    X_data = np.random.default_rng(42).normal(0, 1, (10, 5))
    X = [[float(X_data[i, j]) for j in range(5)] for i in range(10)]
    W_data = np.random.default_rng(42).normal(0, 1, (5, 100))
    W = [[float(W_data[i, j]) for j in range(100)] for i in range(5)]
    result = gcn(A, X, W)
    assert isinstance(result, dict)
    assert "H" in result
