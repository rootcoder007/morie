"""Without music, life would be a mistake. — Friedrich Nietzsche"""

from morie.fn import _array_core as np
from morie.fn.hmfa import geron_flash_attention


def test_hmfa_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    K = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    V = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = geron_flash_attention(Q, K, V)
    assert isinstance(result, dict)
    assert "estimate" in result or "output" in result


def test_hmfa_edge():
    """Test edge cases."""
    Q = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    K = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    V = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = geron_flash_attention(Q, K, V)
    assert isinstance(result, dict)


def test_causal_tiled_attention_recomputed():
    import math

    import pytest

    Q = [[0.3, -0.2], [1.0, 0.5], [0.1, 0.9]]
    K = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    V = [[1.0], [2.0], [4.0]]
    want = []
    for i, q in enumerate(Q):
        s = [sum(a * b for a, b in zip(q, K[j])) / math.sqrt(2) for j in range(i + 1)]
        e = [math.exp(v - max(s)) for v in s]
        want.append(sum(e[j] * V[j][0] for j in range(i + 1)) / sum(e))
    r = geron_flash_attention(Q, K, V, block_size=2, causal=True)
    assert [row[0] for row in r["output"]] == pytest.approx(want, rel=1e-13)
    assert r["peak_score_memory"] == 4 and r["naive_score_memory"] == 9
