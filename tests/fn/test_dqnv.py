"""Tests for dqnv.deep_q_network."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.dqnv import deep_q_network


def _make_chain_mdp(n_states=4, n_actions=2, gamma=0.9):
    """Tiny deterministic MDP used as the test fixture.

    Two actions in every state. Action 0 goes to the next state (or
    stays at the last), action 1 loops back to state 0. Reward is 1
    on the self-loop edge in every state, else 0.
    """
    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))
    for s in range(n_states):
        s_next = min(s + 1, n_states - 1)
        P[s, 0, s_next] = 1.0
        P[s, 1, 0] = 1.0
        R[s, 1] = 1.0
    return P, R


def test_dqnv_basic():
    """Test basic functionality."""
    n_states, n_actions = 4, 2
    P, R = _make_chain_mdp(n_states=n_states, n_actions=n_actions)

    result = deep_q_network(
        P, R, n_states, n_actions,
        gamma=0.9, alpha=0.1,
        steps=2000, C=100, buffer_size=500, batch=8,
        seed=0, use_replay=True, use_target=True,
    )

    # Result must be a mapping with the documented keys.
    assert isinstance(result, dict)
    for key in ("estimate", "Q", "residual_history",
                "final_residual", "greedy_policy",
                "used_replay", "used_target_network",
                "C", "method"):
        assert key in result, f"missing key {key!r} in result"

    # 'estimate' is the canonical output name for this function.
    assert "estimate" in result
    assert "statistic" not in result  # not the key this function returns

    # Shapes must match the MDP dimensions.
    Q = result["estimate"]
    assert len(Q) == n_states
    assert all(len(row) == n_actions for row in Q)

    policy = result["greedy_policy"]
    assert len(policy) == n_states
    assert all(0 <= a < n_actions for a in policy)

    # Bellman residual must be a non-negative number that decreases
    # along the training run (or stays equal at the floor).
    hist = result["residual_history"]
    final_residual = result["final_residual"]
    assert len(hist) >= 1
    assert hist[-1] == final_residual
    for i in range(1, len(hist)):
        assert hist[i] <= hist[i - 1] + 1e-9


def test_dqnv_edge():
    """Test edge cases: tiny MDP, replay and target switches.

    The function must run with the smallest legal state/action counts
    and must accept the optional device flags without raising.
    """
    n_states, n_actions = 1, 1
    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))
    P[0, 0, 0] = 1.0

    result = deep_q_network(
        P, R, n_states, n_actions,
        gamma=0.99, alpha=0.1,
        steps=200, C=20, buffer_size=50, batch=4,
        seed=1, use_replay=True, use_target=True,
    )
    assert isinstance(result, dict)
    assert result["used_replay"] is True
    assert result["used_target_network"] is True
    assert result["C"] == 20
    Q = result["estimate"]
    assert len(Q) == n_states
    assert len(Q[0]) == n_actions

    # Both devices can be turned off; the function still runs.
    result_plain = deep_q_network(
        P, R, n_states, n_actions,
        gamma=0.99, alpha=0.1,
        steps=200, C=20, buffer_size=50, batch=4,
        seed=1, use_replay=False, use_target=False,
    )
    assert isinstance(result_plain, dict)
    assert result_plain["used_replay"] is False
    assert result_plain["used_target_network"] is False

    # Computed Q update must match the vanilla tabular Q-learning
    # rule on the single transition (s=0, a=0 -> s2=0, r=0):
    #   Q[0][0] += alpha * (r + gamma * Q[0][0] - Q[0][0])
    # With Q initialised to 0 this stays at 0 after one update.
    expected_Q00 = 0.0 + 0.1 * (0.0 + 0.99 * 0.0 - 0.0)
    assert abs(result_plain["estimate"][0][0] - expected_Q00) < 1e-9
