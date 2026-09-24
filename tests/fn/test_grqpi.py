"""Tests for grqpi.geron_action_value_function."""

import math

from morie.fn import _array_core as np

from morie.fn.grqpi import geron_action_value_function


def test_grqpi_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    S, A = 3, 2
    # transitions shape (S, A, S); normalize each (s, a) row to sum to 1
    transitions = rng.uniform(0, 1, (S, A, S))
    for s in range(S):
        for a in range(A):
            row = transitions[s][a]
            row_sum = np.sum(row)
            if row_sum > 0:
                transitions[s][a] = [x / row_sum for x in row]
    # rewards must be 3-D (S, A, S) because the underlying pure-Python
    # repeat used by policy_evaluation only supports axis=0 broadcast.
    # Constant per-(s,a) reward, broadcast over next-state axis.
    base_rewards = rng.normal(0, 1, (S, A))
    rewards = [
        [[base_rewards[s][a] for _ in range(S)] for a in range(A)]
        for s in range(S)
    ]
    # deterministic policy shape (S,)
    policy = [int(rng.integers(0, A)) for _ in range(S)]
    state = 0
    action = 0
    gamma = 0.9
    result = geron_action_value_function(
        state, action, policy, transitions, rewards, gamma
    )
    assert isinstance(result, dict)
    assert "q_value" in result
    assert "q_values" in result
    assert "values" in result
    assert "advantage" in result
    assert "greedy_action" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result
    assert math.isfinite(result["q_value"])
    assert math.isfinite(result["advantage"])
    assert result["n"] == S
    assert 0 <= result["greedy_action"] < A
    assert len(result["values"]) == S
    assert len(result["q_values"]) == S
    assert len(result["q_values"][0]) == A


def test_grqpi_edge():
    """Test edge cases."""
    # Single absorbing state, two actions (mirrors the docstring example).
    # rewards is given as 3-D (S, A, S) because the 2-D variant is not
    # supported by the underlying pure-Python repeat.
    transitions = [[[1.0], [1.0]]]
    rewards = [[[1.0], [0.0]]]
    policy = [0]
    state = 0
    action = 1
    gamma = 0.5
    result = geron_action_value_function(
        state, action, policy, transitions, rewards, gamma
    )
    assert isinstance(result, dict)
    assert "q_value" in result
    assert "q_values" in result
    assert "values" in result
    assert "advantage" in result
    assert "greedy_action" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result
    assert math.isfinite(result["q_value"])
    assert math.isfinite(result["advantage"])
    assert result["n"] == 1
    assert 0 <= result["greedy_action"] < 2
    assert len(result["values"]) == 1
    assert len(result["q_values"]) == 1
    assert len(result["q_values"][0]) == 2
