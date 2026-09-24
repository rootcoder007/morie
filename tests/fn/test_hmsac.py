"""Tests for hmsac.geron_sac."""

from morie.fn import _array_core as np

from morie.fn.hmsac import geron_sac


class Bandit:
    """Trivial 1-state, 2-action bandit used as a test environment."""
    n_states, n_actions = 1, 2

    def reset(self):
        return 0

    def step(self, a):
        return 0, float(a), False


def test_hmsac_basic():
    """Test basic functionality."""
    env = Bandit()
    result = geron_sac(env)
    assert isinstance(result, dict)
    for key in ("policy", "Q", "V", "entropy", "returns", "estimate", "n", "method"):
        assert key in result
    assert len(result["policy"]) == env.n_states
    assert len(result["policy"][0]) == env.n_actions


def test_hmsac_edge():
    """Test edge cases."""
    env = Bandit()
    policy = [[0.3, 0.7]]
    critic = [[0.0, 0.0]]
    result = geron_sac(
        env,
        policy=policy,
        critic=critic,
        epochs=5,
        lr=0.5,
        alpha=0.1,
        gamma=0.9,
        steps=5,
        seed=0,
    )
    assert isinstance(result, dict)
    for key in ("policy", "Q", "V", "entropy", "returns", "estimate"):
        assert key in result
    assert len(result["policy"]) == env.n_states
    assert len(result["policy"][0]) == env.n_actions


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmsac as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
