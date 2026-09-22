"""Tests for atq8.int8_attention."""

from morie.fn import _array_core as np

from morie.fn.atq8 import int8_attention


def test_atq8_basic():
    """Test basic functionality."""
    rng_q = np.random.default_rng(42)
    rng_kv = np.random.default_rng(43)
    n_q, n_k, d, d_v = 6, 8, 5, 4
    Q = rng_q.normal(0, 1, (n_q, d))
    K = rng_kv.normal(0, 1, (n_k, d))
    V = rng_kv.normal(0, 1, (n_k, d_v))
    # per-row scales must be strictly positive; build sq, sk, sv from row max-abs
    def row_absmax(M):
        return [max(abs(float(v)) for v in row) for row in M]
    sq = row_absmax(Q)
    sk = row_absmax(K)
    sv = row_absmax(V)
    scales = (sq, sk, sv)

    result = int8_attention(y=None, Q=Q, K=K, V=V, scales=scales)

    # Output must be a dict-like RichResult that supports membership checks
    assert hasattr(result, "__contains__") or isinstance(result, dict)
    assert "output" in result
    assert "scores" in result
    assert "s_q" in result
    assert "s_k" in result
    assert "s_v" in result
    assert "max_abs_error_vs_float" in result

    # Scales are returned as-is (the explicit vectors we provided)
    assert list(result["s_q"]) == sq
    assert list(result["s_k"]) == sk
    assert list(result["s_v"]) == sv

    # Output shape: n_q rows of length d_v
    assert len(result["output"]) == n_q
    for row in result["output"]:
        assert len(row) == d_v

    # Scores shape: n_q rows of length n_k
    assert len(result["scores"]) == n_q
    for row in result["scores"]:
        assert len(row) == n_k

    # Weights form a valid softmax distribution per query row: all non-negative,
    # each row sums to 1.
    weights = result["weights"]
    assert len(weights) == n_q
    for row in weights:
        assert len(row) == n_k
        for w in row:
            assert w >= 0.0
        s = 0.0
        for w in row:
            s += w
        assert abs(s - 1.0) < 1e-9

    # Error must be finite and non-negative
    err = result["max_abs_error_vs_float"]
    assert err >= 0.0


def test_atq8_edge():
    """Test edge cases."""
    rng_q = np.random.default_rng(42)
    rng_kv = np.random.default_rng(43)
    n_q, n_k, d, d_v = 3, 4, 3, 2
    Q = rng_q.normal(0, 1, (n_q, d))
    K = rng_kv.normal(0, 1, (n_k, d))
    V = rng_kv.normal(0, 1, (n_k, d_v))

    def row_absmax(M):
        return [max(abs(float(v)) for v in row) for row in M]
    sq = row_absmax(Q)
    sk = row_absmax(K)
    sv = row_absmax(V)
    scales = (sq, sk, sv)

    result = int8_attention(y=None, Q=Q, K=K, V=V, scales=scales)

    # Even in the "edge" path (still well-formed inputs), output and scores are dict entries
    assert hasattr(result, "__contains__") or isinstance(result, dict)
    assert "output" in result
    assert "scores" in result
    assert "s_q" in result
    assert "s_k" in result
    assert "s_v" in result

    # Output dimensions
    assert len(result["output"]) == n_q
    for row in result["output"]:
        assert len(row) == d_v

    # Scores dimensions
    assert len(result["scores"]) == n_q
    for row in result["scores"]:
        assert len(row) == n_k

    # Dequantised scores are real numbers (not NaN)
    for row in result["scores"]:
        for v in row:
            assert v == v  # NaN check
