"""Tests for atalib.alibi_position_bias."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.atalib import alibi_position_bias


def test_atalib_basic():
    """Test basic functionality with shaped 2D inputs."""
    rng = np.random.default_rng(42)
    n_q, n_k, d, d_v = 3, 4, 5, 2
    Q = rng.normal(0, 1, (n_q, d))
    K = rng.normal(0, 1, (n_k, d))
    V = rng.normal(0, 1, (n_k, d_v))
    slopes = 0.5
    result = alibi_position_bias(Q=Q, K=K, V=V, slopes=slopes, causal=False)

    assert result is not None
    assert isinstance(result, dict)

    # Required payload keys from the function's documented return dict.
    for key in ("output", "weights", "bias", "slopes", "n_q", "n_k", "d", "d_v", "causal", "method"):
        assert key in result

    # One head => output is n_q by d_v matrix (list-of-lists) with right shapes.
    out = result["output"]
    assert isinstance(out, list)
    assert len(out) == n_q
    assert all(len(row) == d_v for row in out)

    # Weights and bias match the documented shapes (n_q by n_k).
    W = result["weights"]
    B = result["bias"]
    assert len(W) == n_q and all(len(r) == n_k for r in W)
    assert len(B) == n_q and all(len(r) == n_k for r in B)

    # The slopes echo and the metadata should match the inputs.
    assert result["slopes"] == [0.5]
    assert result["n_q"] == n_q
    assert result["n_k"] == n_k
    assert result["d"] == d
    assert result["d_v"] == d_v
    assert result["causal"] is False

    # Cross-check the bias: B[i][j] = -m * |i - j| for non-causal.
    m = 0.5
    for i in range(n_q):
        for j in range(n_k):
            expected_ij = -m * abs(i - j)
            assert abs(B[i][j] - expected_ij) < 1e-12

    # Cross-check the first head output against a plain-numpy computation
    # built from the same formula: softmax(QK'/sqrt(d) + B) V.
    sc = 1.0 / (d ** 0.5)
    Qp = Q.tolist()
    Kp = K.tolist()
    Vp = V.tolist()
    Bp = B
    expected_out = [[0.0] * d_v for _ in range(n_q)]
    for i in range(n_q):
        scores = []
        for j in range(n_k):
            dot = 0.0
            for t in range(d):
                dot += Qp[i][t] * Kp[j][t]
            scores.append(dot * sc + Bp[i][j])
        # stable softmax
        mmax = max(scores)
        exps = [pow(2.718281828459045, s - mmax) for s in scores]
        z = sum(exps)
        w = [e / z for e in exps]
        for j in range(n_k):
            for t in range(d_v):
                expected_out[i][t] += w[j] * Vp[j][t]
    for i in range(n_q):
        for t in range(d_v):
            assert abs(out[i][t] - expected_out[i][t]) < 1e-9


def test_atalib_edge():
    """Test edge cases: integer slopes and causal masking."""
    rng = np.random.default_rng(7)
    n_q, n_k, d, d_v = 2, 3, 2, 2
    Q = rng.normal(0, 1, (n_q, d))
    K = rng.normal(0, 1, (n_k, d))
    V = rng.normal(0, 1, (n_k, d_v))

    # Edges: slopes as a length-2 array -> two heads, output becomes a list of matrices.
    slopes = [0.25, 1.0]
    result = alibi_position_bias(Q=Q, K=K, V=V, slopes=slopes, causal=True)

    assert isinstance(result, dict)
    out = result["output"]
    assert isinstance(out, list)
    assert len(out) == 2  # two heads
    for head_out in out:
        assert len(head_out) == n_q
        assert all(len(row) == d_v for row in head_out)

    # Slopes echo.
    assert result["slopes"] == [0.25, 1.0]
    assert result["causal"] is True

    # Under causal=True, the bias for j > i must be a large negative so those
    # keys are effectively masked out (weights ~ 0 for j > i in the first row).
    B = result["bias"]
    for j in range(1, n_k):
        assert B[0][j] <= -1e8

    # With a single-head call, output collapses to a single matrix (not a list).
    one_head = alibi_position_bias(Q=Q, K=K, V=V, slopes=0.5, causal=False)
    out_one = one_head["output"]
    assert isinstance(out_one, list) and len(out_one) == n_q
    assert all(len(row) == d_v for row in out_one)
