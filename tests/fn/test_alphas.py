"""Tests for alphas.alphazero_self_play."""

import math

from morie.fn import _array_core as np

from morie.fn.alphas import alphazero_self_play


def test_alphas_basic():
    """Test basic functionality."""
    state = 0

    def policy(s):
        return ([0.5, 0.3, 0.2], 0.1)

    def step(s, a):
        return (s * 3 + a + 1) % 20

    def terminal(s):
        return s >= 10

    def outcome(s):
        return 1.0

    result = alphazero_self_play(
        state, policy, mcts_iter=4, step=step,
        terminal=terminal, outcome=outcome, max_moves=20,
        temp_threshold=10,
    )

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "states" in result
    assert "pis" in result
    assert "actions" in result
    assert "zs" in result
    assert "moves" in result
    assert "final_state" in result

    moves = result["moves"]
    assert isinstance(moves, int)
    assert moves >= 0
    assert len(result["states"]) == moves
    assert len(result["actions"]) == moves
    assert len(result["zs"]) == moves
    assert math.isfinite(result["estimate"])


def test_alphas_edge():
    """Test edge cases."""
    state = 100  # initial state is already terminal

    def policy(s):
        return ([0.5, 0.3, 0.2], 0.0)

    def terminal(s):
        return s >= 100

    def outcome(s):
        return 0.5

    result = alphazero_self_play(
        state, policy, mcts_iter=2, terminal=terminal,
        outcome=outcome, max_moves=10,
    )

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "moves" in result
    assert result["moves"] == 0
    assert math.isfinite(result["estimate"])
