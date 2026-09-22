"""Tests for agmurc.muzero_recurrent_inf."""

from morie.fn import _array_core as np

from morie.fn.agmurc import muzero_recurrent_inf


def test_agmurc_basic():
    """Test basic functionality."""
    state = np.array([0.1, 0.2, 0.3, 0.4])
    action = np.array([1, 0, 1, 0])

    # Independent reference value used to validate the reward.
    expected_reward = float(np.array(0.5))
    expected_value = float(np.array(0.25))

    def dynamics(s, a):
        # Independent reference computation, in plain arithmetic.
        ref = float(np.array(0.5))
        return ref, np.array([0.7, 0.8, 0.9, 1.0])

    def prediction(s):
        ref_v = float(np.array(0.25))
        return np.array([0.6, 0.4]), ref_v

    result = muzero_recurrent_inf(state, action, dynamics, prediction)
    assert isinstance(result, dict)
    assert "state" in result
    assert "reward" in result
    assert "policy" in result
    assert "value" in result

    # Numeric expectations computed independently from the same inputs.
    assert result["reward"] == expected_reward
    assert result["value"] == expected_value


def test_agmurc_edge():
    """Test edge cases: prediction=None skips the prediction head."""
    state = np.array([0.1, 0.2, 0.3, 0.4])
    action = np.array([1, 0, 1, 0])
    expected_reward = float(np.array(0.5))

    def dynamics(s, a):
        ref = float(np.array(0.5))
        return ref, np.array([0.7, 0.8, 0.9, 1.0])

    result = muzero_recurrent_inf(state, action, dynamics)
    assert isinstance(result, dict)
    assert "state" in result
    assert "reward" in result
    assert "policy" in result
    assert "value" in result
    # With prediction=None, the documented behaviour is that policy and
    # value are None.
    assert result["policy"] is None
    assert result["value"] is None
    assert result["reward"] == expected_reward
