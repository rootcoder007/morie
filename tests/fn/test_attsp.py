"""Tests for attsp.sparse_attention."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.attsp import sparse_attention


def _softmax(x, allow):
    mx = max(x[j] for j in allow)
    e = [0.0] * len(x)
    tot = 0.0
    for j in allow:
        e[j] = np.exp(x[j] - mx)
        tot += e[j]
    return [v / tot for v in e]


def test_attsp_basic():
    """Test basic functionality with a hand-computable pattern."""
    d_dim = 4
    Q = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])
    K = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    V = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
    ])
    S = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
    ])
    result = sparse_attention(Q, K, V, S)

    # Returned object must be a dict-like mapping (RichResult supports .__contains__).
    assert "out" in result
    assert "weight" in result
    assert "score" in result
    assert "density" in result
    assert "nq" in result and result["nq"] == 2
    assert "nk" in result and result["nk"] == 4
    assert "d" in result and result["d"] == 4
    assert "dv" in result and result["dv"] == 2
    assert result["density"] == 0.5

    # Independent computation of the documented formula.
    sc = np.sqrt(d_dim)
    sco = [[sum(Q[i][t] * K[j][t] for t in range(d_dim)) / sc
            for j in range(4)] for i in range(2)]
    for i in range(2):
        for j in range(4):
            assert abs(result["score"][i][j] - sco[i][j]) < 1e-9

    allow = [[0, 2], [1, 3]]
    Wt = [_softmax(sco[i], allow[i]) for i in range(2)]
    for i in range(2):
        for j in range(4):
            assert abs(result["weight"][i][j] - Wt[i][j]) < 1e-9

    expected_out = [[sum(Wt[i][j] * V[j][t] for j in range(4)) for t in range(2)]
                    for i in range(2)]
    for i in range(2):
        for t in range(2):
            assert abs(result["out"][i][t] - expected_out[i][t]) < 1e-9


def test_attsp_edge():
    """Test the dense (S=None) edge case."""
    Q = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    K = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    V = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = sparse_attention(Q, K, V)
    assert "out" in result
    assert "density" in result
    assert result["density"] == 1.0
    # Every row of the weight matrix sums to 1.
    for row in result["weight"]:
        assert abs(sum(row) - 1.0) < 1e-9
