"""Tests for agmuzu.muzero_world_model."""

from morie.fn import _array_core as np

from morie.fn.agmuzu import muzero_world_model


def test_agmuzu_basic():
    """Test basic functionality."""
    rng_obs = np.random.default_rng(42)
    rng_dyn = np.random.default_rng(43)
    observations = rng_obs.normal(0, 1, 100)

    K = 3

    def representation(_obs):
        return 0.0

    def dynamics(state, action):
        return state + action, state + action

    actions = [1.0, 2.0, 3.0]

    result = muzero_world_model(observations, actions, representation, dynamics)

    assert isinstance(result, dict)
    assert "states" in result
    assert "rewards" in result
    assert "root" in result
    assert "K" in result

    assert len(result["states"]) == K + 1
    assert len(result["rewards"]) == K
    assert result["K"] == K

    expected_root = 0.0
    assert result["root"] == expected_root

    expected_rewards = [0.0 + 1.0, 1.0 + 2.0, 3.0 + 3.0]
    assert list(result["rewards"]) == expected_rewards

    expected_states = [0.0]
    s = 0.0
    for a in actions:
        s = s + a
        expected_states.append(s)
    assert list(result["states"]) == expected_states


def test_agmuzu_edge():
    """Test edge cases."""
    rng_obs = np.random.default_rng(42)
    observations = rng_obs.normal(0, 1, 100)

    def representation(_obs):
        return "root_state"

    def dynamics(state, action):
        return action, state + "_" + str(action)

    actions = []

    result = muzero_world_model(observations, actions, representation, dynamics)

    assert isinstance(result, dict)
    assert result["K"] == 0
    assert len(result["states"]) == 1
    assert len(result["rewards"]) == 0
    assert result["root"] == "root_state"
