"""Tests for gatV2.gat_v2."""

from morie.fn import _array_core as np

from morie.fn.gatV2 import gat_v2


def test_gatV2_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 4
    p = 3

    # Build a square adjacency matrix as nested list of scalars (10x10 -> here 4x4 for tractable test).
    A = [[float(v) for v in row] for row in rng.normal(0.0, 1.0, (n, n))]

    # X must have one row per node (n rows) with p feature columns.
    X = [[float(v) for v in row] for row in rng.normal(0.0, 1.0, (n, p))]

    result = gat_v2(A, X)

    # Documented return is a RichResult with a payload dict.
    assert hasattr(result, "payload")
    payload = result.payload
    assert "estimate" in payload
    assert "H" in payload
    assert "alpha_first" in payload
    assert "n" in payload
    assert "method" in payload

    assert payload["n"] == n

    # H must be n x p after the sigmoid.
    H = payload["H"]
    assert len(H) == n
    assert all(len(row) == p for row in H)
    # All values must be in (0, 1) due to sigmoid.
    for row in H:
        for v in row:
            assert 0.0 < v < 1.0

    # alpha_first: attention weights for node 0, softmax over (self + nonzero neighbours).
    # By definition, sum of softmax weights must be 1.
    alpha_first = payload["alpha_first"]
    assert abs(sum(alpha_first) - 1.0) < 1e-9
    assert all(0.0 <= w <= 1.0 for w in alpha_first)

    # Independent computation of the estimate: mean of the sigmoid-weighted sum
    # of feature rows selected by softmax over (self + neighbours of node 0).
    # LeakyReLU(x) = x if x >= 0 else 0.2 * x, using the documented default 0.2.
    def lrelu(v):
        return v if v >= 0.0 else 0.2 * v

    # Reconstruct X as plain Python lists.
    X_list = X
    n_nodes = len(X_list)
    feat = len(X_list[0])

    g = [sum(lrelu(X_list[i][c]) for c in range(feat)) for i in range(n_nodes)]

    i = 0
    nb = [j for j in range(n_nodes) if j == i or A[i][j] != 0.0]
    e = [g[i] + g[j] for j in nb]
    mx = max(e)
    # stable softmax
    w = [math_exp(v - mx) for v in e]
    tot = sum(w)
    w = [v / tot for v in w]

    def sigmoid(v):
        if v >= 0.0:
            z = math_exp(-v)
            return 1.0 / (1.0 + z)
        else:
            z = math_exp(v)
            return z / (1.0 + z)

    # For each feature column c, compute sum_j w[j] * X_list[nb[j]][c], then sigmoid.
    H_node0 = []
    for c in range(feat):
        s = 0.0
        for a_idx, j in enumerate(nb):
            s += w[a_idx] * X_list[j][c]
        H_node0.append(sigmoid(s))

    # The independent estimate is the mean across all nodes and features; we check
    # that node 0's row matches our hand-computed values.
    for c in range(feat):
        assert abs(H[0][c] - H_node0[c]) < 1e-9

    # Independently compute the global estimate as mean of all H entries.
    flat = [v for row in H for v in row]
    expected_estimate = sum(flat) / len(flat)
    assert abs(payload["estimate"] - expected_estimate) < 1e-12

    assert "GATv2" in payload["method"]


# Local helpers using math to avoid relying on the implementation under test.
import math as _math


def math_exp(x):
    return _math.exp(x)


def test_gatV2_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 4
    p = 5

    # Build a square adjacency matrix with a diagonal (self-loops) so every node
    # has at least itself as a neighbour.
    A = [[float(v) for v in row] for row in rng.normal(0.0, 1.0, (n, n))]
    # Ensure diagonal entries are non-zero so the "j == i or M[i][j] != 0.0"
    # neighbour rule always includes the node itself.
    for i in range(n):
        A[i][i] = 1.0

    X = [[float(v) for v in row] for row in rng.normal(0.0, 1.0, (n, p))]

    result = gat_v2(A, X)
    assert hasattr(result, "payload")
    payload = result.payload
    assert isinstance(payload["H"], list)
    assert len(payload["H"]) == n
    assert payload["n"] == n
    # With self-loops every node has exactly n neighbours (all nodes),
    # so alpha_first must have exactly n entries and sum to 1.
    assert len(payload["alpha_first"]) == n
    assert abs(sum(payload["alpha_first"]) - 1.0) < 1e-9
