"""Tests for hmmTag.hmm_pos."""

import math

from morie.fn import _array_core as np
from morie.fn.hmmTag import hmm_pos


def test_hmmTag_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    S = 3
    V = 5
    n = 40
    X = rng.integers(0, V, n).tolist()
    tagset = [f"tag{i}" for i in range(S)]
    start = [0.5, 0.3, 0.2]
    trans = [[0.6, 0.3, 0.1],
             [0.2, 0.6, 0.2],
             [0.1, 0.3, 0.6]]
    emit = [[0.2, 0.2, 0.2, 0.2, 0.2],
            [0.3, 0.3, 0.2, 0.1, 0.1],
            [0.1, 0.1, 0.3, 0.3, 0.2]]
    result = hmm_pos(X, tagset, start=start, trans=trans, emit=emit)
    assert isinstance(result, dict)
    assert "path" in result
    assert "logprob" in result
    assert "n" in result
    assert "estimate" in result
    assert result["n"] == n
    assert len(result["path"]) == n
    assert math.isfinite(result["logprob"])
    assert all(1 <= p <= S for p in result["path"])


def test_hmmTag_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    S = 2
    V = 4
    n = 5
    X = rng.integers(0, V, n).tolist()
    tagset = ["N", "V"]
    start = [0.6, 0.4]
    trans = [[0.7, 0.3],
             [0.4, 0.6]]
    emit = [[0.25, 0.25, 0.25, 0.25],
            [0.1, 0.4, 0.4, 0.1]]
    result = hmm_pos(X, tagset, start=start, trans=trans, emit=emit)
    assert isinstance(result, dict)
    assert "path" in result
    assert result["n"] == n
    assert len(result["path"]) == n
    assert math.isfinite(result["logprob"])
    assert all(1 <= p <= S for p in result["path"])
