"""Tests for hmppo.geron_ppo."""

from morie.fn import _array_core as np

from morie.fn.hmppo import geron_ppo


def test_hmppo_basic():
    """Test basic functionality."""
    # Simple 2-state, 2-action environment
    state = [0]

    def reset():
        state[0] = 0
        return 0

    def step(a):
        reward = 1.0 if a == state[0] else 0.0
        return state[0], reward, True

    env = {"reset": reset, "step": step}

    rng = np.random.default_rng(42)
    # Policy: (n_states, n_actions) = (2, 2) logit matrix
    policy = rng.normal(0, 1, (2, 2))

    result = geron_ppo(
        env, policy,
        epochs=5, lr=0.1, clip_eps=0.2,
        n_episodes=4, max_steps=10, n_updates=2, seed=0,
    )

    assert isinstance(result, dict)
    for key in ["theta", "probabilities", "return_history", "surrogate_history",
                "clip_fraction", "estimate", "n", "method"]:
        assert key in result

    # clip_fraction is a fraction in [0, 1]
    assert 0.0 <= result["clip_fraction"] <= 1.0
    # theta and probabilities should match the policy shape (2, 2)
    assert len(result["theta"]) == 2
    assert len(result["theta"][0]) == 2
    assert len(result["probabilities"]) == 2
    assert len(result["probabilities"][0]) == 2
    # return_history and surrogate_history are per-epoch sequences
    assert len(result["return_history"]) == 5
    assert len(result["surrogate_history"]) == 5
    # estimate is a per-state array matching the policy shape
    est = result["estimate"]
    assert len(est) == 2
    assert len(est[0]) == 2
    # n is a positive int
    assert isinstance(result["n"], int) and result["n"] > 0


def test_hmppo_edge():
    """Test edge cases."""
    # Minimal one-state, two-action bandit (matches the docstring example shape)
    def reset():
        return 0

    def step(a):
        return 0, float(a), True

    env = {"reset": reset, "step": step}

    # Single state, 2 actions
    policy = [[0.0, 0.0]]

    result = geron_ppo(
        env, policy,
        epochs=3, lr=0.1, clip_eps=0.2,
        n_episodes=4, max_steps=5, n_updates=2, seed=1,
    )

    assert isinstance(result, dict)
    for key in ["theta", "probabilities", "return_history", "surrogate_history",
                "clip_fraction", "estimate", "n", "method"]:
        assert key in result

    # clip_fraction is a fraction in [0, 1]
    assert 0.0 <= result["clip_fraction"] <= 1.0
    # theta and probabilities should match the policy shape (1, 2)
    assert len(result["theta"]) == 1
    assert len(result["theta"][0]) == 2
    assert len(result["probabilities"]) == 1
    assert len(result["probabilities"][0]) == 2
    # Probabilities of a softmax must be nonnegative and sum to 1 per state
    row = result["probabilities"][0]
    assert all(p >= 0.0 for p in row)
    s = 0.0
    for p in row:
        s += p
    assert abs(s - 1.0) < 1e-6
    # Per-epoch histories match the epoch count
    assert len(result["return_history"]) == 3
    assert len(result["surrogate_history"]) == 3
    # estimate is a per-state array matching the policy shape
    est = result["estimate"]
    assert len(est) == 1
    assert len(est[0]) == 2


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmppo as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
