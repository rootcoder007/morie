"""Tests for hmddpg.geron_ddpg."""

from morie.fn import _array_core as np
import math

from morie.fn.hmddpg import geron_ddpg


def test_hmddpg_basic():
    """Test basic functionality."""
    def env(s, a):
        # 1-D environment with reward -(a - 1)^2, state passes through
        return s, -float((a - 1.0) ** 2), False

    d = 1
    actor = [0.0]            # shape (d,)
    critic = [0.0, 0.0]      # shape (d + 1,) over [s, a]

    result = geron_ddpg(env, actor, critic, epochs=20, lr=0.05,
                        ou_sigma=0.1, seed=1)

    # RichResult behaves like a dict
    assert isinstance(result, dict)

    # Keys named in the return statement
    for key in ("actor", "critic", "actor_target", "critic_target",
                "critic_losses", "rewards", "actions", "ou_noise",
                "q_values", "estimate", "n", "method"):
        assert key in result

    # Per-epoch traces have length == epochs
    epochs = 20
    assert len(result["rewards"]) == epochs
    assert len(result["critic_losses"]) == epochs
    assert len(result["actions"]) == epochs
    assert len(result["ou_noise"]) == epochs
    assert len(result["q_values"]) == epochs

    # The aggregate estimate is a finite number
    assert math.isfinite(float(result["estimate"]))
    assert int(result["n"]) == epochs


def test_hmddpg_edge():
    """Test edge cases."""
    def env(s, a):
        # Trivial zero-reward, state-passthrough environment
        return s, 0.0, False

    d = 1
    actor = [0.0]
    critic = [0.0, 0.0]

    # Minimum-size run: one epoch, low learning rate, deterministic seed
    result = geron_ddpg(env, actor, critic, epochs=1, lr=0.0,
                        gamma=0.0, tau=1.0, ou_theta=0.0, ou_sigma=0.0,
                        seed=0)

    assert isinstance(result, dict)
    assert "rewards" in result
    assert len(result["rewards"]) == 1
    assert len(result["critic_losses"]) == 1
    assert math.isfinite(float(result["estimate"]))


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmddpg as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
