"""Tests for moelyr.moe_layer."""

import math

from morie.fn import _array_core as np

from morie.fn.moelyr import moe_layer


def test_moelyr_basic():
    """Test basic functionality with a small MoE layer."""
    rng = np.random.default_rng(42)
    n = 10  # input dimension
    m = 5   # number of experts
    d = 3   # output dimension per expert
    x = rng.normal(0, 1, n)             # input vector
    W_g = rng.normal(0, 1, (n, m))      # gate weights (n_inputs x n_experts)
    experts = rng.normal(0, 1, (m, d))  # expert outputs (n_experts x n_outputs)
    top_k = 2
    result = moe_layer(x, W_g=W_g, experts=experts, top_k=top_k)
    # Payload keys as documented in the return statement
    assert "estimate" in result
    assert "out" in result
    assert "gate" in result
    assert "chosen" in result
    assert "h" in result
    assert "keep" in result
    # Structural / shape checks
    assert len(result["out"]) == d
    assert len(result["gate"]) == m
    assert len(result["h"]) == m
    assert len(result["keep"]) == m
    assert len(result["chosen"]) == top_k
    # Chosen experts are distinct and in range
    chosen = result["chosen"]
    assert all(0 <= i < m for i in chosen)
    assert len(set(chosen)) == top_k
    # Softmax weights over the kept experts sum to 1
    assert math.isclose(sum(result["gate"]), 1.0, abs_tol=1e-6)
    # Non-chosen experts receive zero gate weight
    for i in range(m):
        if i not in chosen:
            assert result["gate"][i] == 0.0


def test_moelyr_edge():
    """Test edge case: top_k equal to the number of experts (no sparsity)."""
    rng = np.random.default_rng(43)
    n = 8
    m = 4
    d = 2
    x = rng.normal(0, 1, n)
    W_g = rng.normal(0, 1, (n, m))
    experts = rng.normal(0, 1, (m, d))
    # top_k = m: every expert is kept
    result = moe_layer(x, W_g=W_g, experts=experts, top_k=m)
    assert "estimate" in result
    assert "out" in result
    assert "gate" in result
    assert "chosen" in result
    assert len(result["chosen"]) == m
    assert sorted(result["chosen"]) == list(range(m))
    assert len(result["gate"]) == m
    assert len(result["out"]) == d
    # All gate weights are nonnegative and sum to 1
    assert all(g >= 0 for g in result["gate"])
    assert math.isclose(sum(result["gate"]), 1.0, abs_tol=1e-6)
