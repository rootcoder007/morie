"""Tests for agpuct.alphazero_puct."""

import math

from morie.fn import _array_core as np
from morie.fn.agpuct import alphazero_puct


def test_agpuct_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    k = 5
    P = rng.uniform(0, 1, k)
    s = float(np.sum(P))
    P = [v / s for v in P]
    N = [float(v) for v in rng.integers(0, 20, k)]
    Q = rng.normal(0, 1, k)
    result = alphazero_puct(P, N, Q, c_puct=1.5)
    assert isinstance(result, dict)
    for key in ("score", "estimate", "U", "action",
                "n_parent", "sqrt_n_parent", "c_puct",
                "k", "method"):
        assert key in result
    assert result["k"] == k
    assert len(result["score"]) == k
    assert len(result["U"]) == k
    assert math.isfinite(result["estimate"])
    assert 0 <= result["action"] < k
    assert result["c_puct"] == 1.5


def test_agpuct_edge():
    """Test edge case: zero visit counts and zero exploration constant."""
    k = 3
    P = [1.0 / k] * k
    N = [0.0, 0.0, 0.0]
    Q = [0.5, -0.3, 0.1]
    result = alphazero_puct(P, N, Q, c_puct=0.0)
    assert isinstance(result, dict)
    assert result["k"] == k
    assert result["c_puct"] == 0.0
    assert result["n_parent"] == 0.0
    assert result["sqrt_n_parent"] == 0.0
    assert len(result["score"]) == k
    assert len(result["U"]) == k
    assert all(u == 0.0 for u in result["U"])
    assert result["action"] == 0
    assert math.isfinite(result["estimate"])
