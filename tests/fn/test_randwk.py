"""Tests for randwk.random_walk."""

from morie.fn import _array_core as np

from morie.fn.randwk import random_walk


def test_randwk_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 10
    G = rng.uniform(0, 1, (n, n))
    start = 3
    steps = 5
    result = random_walk(G, start, steps)
    assert isinstance(result, dict)
    assert "p" in result
    assert "estimate" in result
    assert "argmax" in result
    assert "p_start" in result
    assert "n" in result
    assert result["n"] == n
    assert len(result["p"]) == n
    # p is a probability distribution (rows of P sum to 1, so total is preserved)
    total = 0.0
    for pi in result["p"]:
        total += pi
        assert pi >= 0.0
    assert abs(total - 1.0) < 1e-9
    # p_start is the return probability, in [0, 1]
    assert result["p_start"] >= 0.0
    assert result["p_start"] <= 1.0
    # estimate is the largest entry of p
    assert result["estimate"] == max(result["p"])
    # argmax is the index of the largest entry of p
    am = 0
    for i in range(1, n):
        if result["p"][i] > result["p"][am]:
            am = i
    assert result["argmax"] == am


def test_randwk_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 5
    G = np.eye(n)
    start = 2
    steps = 0
    result = random_walk(G, start, steps)
    assert isinstance(result, dict)
    assert result["n"] == n
    # With zero steps, the distribution is concentrated at start
    assert result["p"][start] == 1.0
    for i in range(n):
        if i != start:
            assert result["p"][i] == 0.0
    assert result["estimate"] == 1.0
    assert result["argmax"] == start
    assert result["p_start"] == 1.0
