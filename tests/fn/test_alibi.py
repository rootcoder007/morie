"""Tests for alibi.alibi."""

from morie.fn import _array_core as np

from morie.fn.alibi import alibi


def test_alibi_basic():
    """Test basic functionality with a 2-D score matrix and default slope."""
    rng = np.random.default_rng(42)
    scores = rng.uniform(0, 1, (6, 8))
    result = alibi(scores)
    assert isinstance(result, dict)
    assert "biased" in result
    assert "estimate" in result
    assert "bias" in result
    assert "slope" in result
    assert "n_q" in result
    assert "n_k" in result
    assert "causal" in result
    assert result["n_q"] == 6
    assert result["n_k"] == 8
    assert result["slope"] == 2.0 ** -8.0
    assert result["causal"] is False
    assert len(result["biased"]) == 6
    assert len(result["biased"][0]) == 8
    assert len(result["bias"]) == 6
    assert len(result["bias"][0]) == 8


def test_alibi_edge():
    """Test edge cases with explicit slope and causal masking on a small matrix."""
    rng = np.random.default_rng(7)
    scores = rng.uniform(-1, 1, (3, 4))
    result = alibi(scores, slopes=0.25, causal=True)
    assert isinstance(result, dict)
    assert result["n_q"] == 3
    assert result["n_k"] == 4
    assert result["slope"] == 0.25
    assert result["causal"] is True
    # causal mask should set entries below the diagonal (j > i) to -inf in bias
    bias = result["bias"]
    for i in range(3):
        for j in range(4):
            if j > i:
                assert bias[i][j] == float("-inf")
            else:
                # bias[i][j] == -slope * |i - j|
                assert bias[i][j] == -0.25 * abs(i - j)
    # biased[i][j] for j > i should also be -inf
    biased = result["biased"]
    for i in range(3):
        for j in range(4):
            if j > i:
                assert biased[i][j] == float("-inf")
            else:
                assert biased[i][j] == scores[i][j] - 0.25 * abs(i - j)
