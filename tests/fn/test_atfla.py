"""I cannot teach anybody anything. I can only make them think. — Socrates"""

from morie.fn import _array_core as np

from morie.fn.atfla import flash_attention_block


def test_atfla_basic():
    """Test basic functionality with a hand-traceable example."""
    import math

    # n_q = 2, d = 2; n_k = 3, d_v = 2 (so K is 3x2, V is 3x2)
    Q = [[1.0, 0.0],
         [0.0, 1.0]]
    K = [[1.0, 0.0],
         [0.0, 1.0],
         [1.0, 1.0]]
    V = [[1.0, 2.0],
         [3.0, 4.0],
         [5.0, 6.0]]
    block_size = 2

    result = flash_attention_block(Q=Q, K=K, V=V, block_size=block_size)

    # RichResult: payload is a dict-like
    assert "output" in result
    assert "l" in result
    assert "m" in result
    assert "n_blocks" in result
    assert "estimate" in result

    # Independent computation via the documented formula:
    # scores_ij = <Q_i, K_j> / sqrt(d), d=2 -> scale = 1/sqrt(2)
    sc = 1.0 / math.sqrt(2)
    n_q = len(Q)
    n_k = len(K)
    d = len(Q[0])
    assert d == 2

    # Full softmax (exact reference, no tiling), no causal masking
    output_expected = [[0.0] * 2 for _ in range(n_q)]
    for i in range(n_q):
        scores = []
        for j in range(n_k):
            dot = sum(Q[i][t] * K[j][t] for t in range(d))
            scores.append(dot * sc)
        m_i = max(scores)
        exps = [math.exp(s - m_i) for s in scores]
        l_i = sum(exps)
        for t in range(2):
            acc = 0.0
            for j in range(n_k):
                acc += exps[j] * V[j][t]
            output_expected[i][t] = acc / l_i

    out = result["output"]
    for i in range(n_q):
        for t in range(2):
            assert abs(out[i][t] - output_expected[i][t]) < 1e-9, (
                f"output[{i}][{t}] = {out[i][t]} vs expected {output_expected[i][t]}"
            )

    # n_blocks: ceil(3 / 2) = 2
    assert result["n_blocks"] == 2

    # estimate is output[0][0], independent check
    assert abs(result["estimate"] - output_expected[0][0]) < 1e-9


def test_atfla_edge():
    """Test edge cases: single block and causal masking consistency."""
    import math

    # Single block: block_size >= n_k, so one tile, no rescaling needed
    Q = [[1.0, 0.0],
         [1.0, 1.0]]
    K = [[1.0, 0.0],
         [0.0, 1.0]]
    V = [[1.0, 2.0],
         [3.0, 4.0]]
    block_size = 5

    result = flash_attention_block(Q=Q, K=K, V=V, block_size=block_size)

    assert "output" in result
    assert result["n_blocks"] == 1

    # Independent computation
    sc = 1.0 / math.sqrt(2)
    output_expected = [[0.0] * 2 for _ in range(2)]
    for i in range(2):
        scores = []
        for j in range(2):
            dot = sum(Q[i][t] * K[j][t] for t in range(2))
            scores.append(dot * sc)
        m_i = max(scores)
        exps = [math.exp(s - m_i) for s in scores]
        l_i = sum(exps)
        for t in range(2):
            acc = 0.0
            for j in range(2):
                acc += exps[j] * V[j][t]
            output_expected[i][t] = acc / l_i

    out = result["output"]
    for i in range(2):
        for t in range(2):
            assert abs(out[i][t] - output_expected[i][t]) < 1e-9
