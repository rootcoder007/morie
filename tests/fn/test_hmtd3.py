"""Tests for hmtd3.geron_td3."""

from morie.fn import _array_core as np

from morie.fn.hmtd3 import geron_td3


def test_hmtd3_basic():
    """Test basic functionality."""

    class Bandit:
        n_states, n_actions = 1, 2

        def reset(self):
            return 0

        def step(self, a):
            return 0, float(a), False

    rng = np.random.default_rng(42)
    Q1 = rng.normal(0, 1, (1, 2))
    Q2 = rng.normal(0, 1, (1, 2))
    policy = np.array([0])
    result = geron_td3(
        Bandit(),
        policy=policy,
        Q1=Q1,
        Q2=Q2,
        epochs=40,
        lr=0.5,
        gamma=0.9,
        steps=20,
        policy_delay=2,
        tau=0.5,
        noise=0.2,
        seed=42,
    )
    assert isinstance(result, dict)
    assert "policy" in result
    assert "Q1" in result
    assert "Q2" in result
    assert "returns" in result
    assert "overestimation_gap" in result
    assert "policy_updates" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result
    assert result["overestimation_gap"] >= 0.0
    assert result["policy_updates"] >= 0
    assert len(result["policy"]) == 1


def test_hmtd3_edge():
    """Test edge cases."""

    class Bandit:
        n_states, n_actions = 1, 2

        def reset(self):
            return 0

        def step(self, a):
            return 0, float(a), False

    # Minimal valid configuration with the smallest legal epoch count
    result = geron_td3(
        Bandit(),
        epochs=2,
        steps=2,
        policy_delay=1,
        lr=0.5,
        gamma=0.5,
        tau=0.5,
        noise=0.0,
        seed=0,
    )
    assert isinstance(result, dict)
    assert "method" in result
    assert "n" in result
    assert "policy" in result
    assert "estimate" in result
    assert len(result["policy"]) == 1


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmtd3 as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
