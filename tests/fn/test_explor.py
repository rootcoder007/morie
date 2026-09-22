"""Tests for explor.intrinsic_motivation."""

from morie.fn import _array_core as np

from morie.fn.explor import intrinsic_motivation


def _make_batch(T=20, d=4, n_actions=3, seed=0):
    rng = np.random.default_rng(seed)
    states = rng.normal(0.0, 1.0, (T, d))
    next_states = states + rng.normal(0.0, 0.1, (T, d))
    actions = rng.integers(0, n_actions, T)
    return states, actions, next_states, n_actions


def test_explor_basic():
    """Test basic functionality on a small batch of transitions."""
    states, actions, next_states, n_actions = _make_batch(
        T=20, d=4, n_actions=3, seed=0
    )
    beta = 0.2
    eta = 1.0
    result = intrinsic_motivation(
        states,
        actions,
        next_states,
        n_actions=n_actions,
        n_features=8,
        eta=eta,
        beta=beta,
        lr=0.05,
        epochs=1,
        features="inverse",
        discrete=True,
        seed=0,
    )
    # The function returns a mapping exposing the documented keys.
    assert isinstance(result, dict)
    assert "estimate" in result
    # eq. 6: r^i_t == eta * L_F (per-transition).
    rewards = np.asarray(result["estimate"])
    assert len(rewards) == len(states)
    assert float(np.min(rewards)) >= 0.0
    # forward_loss is the mean L_F, so eta * L_F equals eta * mean(L_F).
    expected_mean = float(eta) * float(result["forward_loss"])
    got_mean = float(np.mean(rewards))
    assert abs(got_mean - expected_mean) < 1e-9
    # beta is in [0, 1]; the objective is (1-beta) L_I + beta L_F.
    expected_obj = (1.0 - beta) * float(result["inverse_loss"]) \
        + beta * float(result["forward_loss"])
    assert abs(float(result["objective"]) - expected_obj) < 1e-9


def test_explor_edge():
    """Edge case: a minimal two-transition batch with both action classes."""
    states, actions, next_states, n_actions = _make_batch(
        T=2, d=4, n_actions=2, seed=1
    )
    result = intrinsic_motivation(
        states,
        actions,
        next_states,
        n_actions=n_actions,
        n_features=4,
        eta=1.0,
        beta=0.0,
        epochs=1,
        features="inverse",
        discrete=True,
        seed=1,
    )
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "loss_curve" in result
    assert len(result["estimate"]) == 2
    assert len(result["loss_curve"]) == 1
    # beta=0 means the objective reduces to L_I only.
    assert abs(
        float(result["objective"]) - float(result["inverse_loss"])
    ) < 1e-12
