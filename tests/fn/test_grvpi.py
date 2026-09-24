"""Tests for grvpi.geron_state_value_function."""

import math

from morie.fn import _array_core as np

from morie.fn.grvpi import geron_state_value_function


def _make_transitions(rng, n_states, n_actions):
    """Build a (S, A, S) transition tensor whose rows sum to 1."""
    out = []
    for _ in range(n_states):
        block = []
        for _ in range(n_actions):
            row = [abs(x) for x in rng.normal(0, 1, n_states)]
            total = sum(row)
            block.append([x / total for x in row])
        out.append(block)
    return out


def test_grvpi_basic():
    """Basic random MDP: returns a RichResult with the documented keys."""
    rng = np.random.default_rng(42)
    n_states, n_actions = 3, 2

    transitions = _make_transitions(rng, n_states, n_actions)
    rewards = [
        [list(rng.normal(0, 1, n_states)) for _ in range(n_actions)]
        for _ in range(n_states)
    ]
    policy = [0, 1, 0]  # (S,) deterministic actions
    state = 0
    gamma = 0.9

    result = geron_state_value_function(state, policy, transitions, rewards, gamma)

    assert isinstance(result, dict)
    for key in ("value", "values", "state", "estimate", "n", "method"):
        assert key in result
    assert result["state"] == state
    assert result["n"] == n_states
    assert len(result["values"]) == n_states
    assert math.isfinite(result["value"])
    assert math.isfinite(result["estimate"])
    for v in result["values"]:
        assert math.isfinite(v)


def test_grvpi_edge():
    """Edge case: single absorbing state with constant reward 1 (docstring example)."""
    # Absorbing MDP, value = 1 / (1 - gamma) = 2.0 when gamma = 0.5
    transitions = [[[1.0]]]  # shape (1, 1, 1)
    rewards = [[[1.0]]]      # shape (1, 1, 1)
    policy = [0]              # shape (1,)
    state = 0
    gamma = 0.5

    result = geron_state_value_function(state, policy, transitions, rewards, gamma)

    assert isinstance(result, dict)
    for key in ("value", "values", "state", "estimate", "n", "method"):
        assert key in result
    assert result["state"] == 0
    assert result["n"] == 1
    assert len(result["values"]) == 1
    assert abs(result["value"] - 2.0) < 1e-9
    assert abs(result["values"][0] - 2.0) < 1e-9
