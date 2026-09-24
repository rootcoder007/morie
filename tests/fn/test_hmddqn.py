"""Tests for hmddqn.geron_double_dqn."""

import math

from morie.fn import _array_core as np

from morie.fn.hmddqn import geron_double_dqn


def test_hmddqn_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    S, A = 3, 2
    Q = rng.normal(0, 1, (S, A))
    Q_target = rng.normal(0, 1, (S, A))
    # buffer entries: (state, action, reward, next_state, done)
    buffer = [
        (0, 0, 0.5, 1, False),
        (1, 1, -0.5, 2, False),
        (2, 0, 1.0, 0, False),
    ]
    result = geron_double_dqn(
        None, Q, Q_target, buffer, epochs=5, lr=0.1, gamma=0.9,
    )
    assert isinstance(result, dict)
    for key in (
        "Q", "Q_target", "loss_history", "targets", "vanilla_targets",
        "overestimation_gap", "greedy_policy", "sync_epochs",
        "estimate", "n", "method",
    ):
        assert key in result
    Qr = np.array(result["Q"])
    assert Qr.shape == (S, A)
    Qtr = np.array(result["Q_target"])
    assert Qtr.shape == (S, A)
    assert len(result["targets"]) == len(buffer)
    assert len(result["vanilla_targets"]) == len(buffer)
    assert len(result["overestimation_gap"]) == len(buffer)
    assert math.isfinite(float(result["estimate"]))
    assert int(result["n"]) == len(buffer)


def test_hmddqn_edge():
    """Test edge cases: terminal transition removes the bootstrap."""
    # Reproduces the second worked example from the docstring.
    Q = [[0.0, 1.0]]
    Q_target = [[10.0, -10.0]]
    buffer = [(0, 0, 5.0, 0, True)]
    result = geron_double_dqn(
        None, Q, Q_target, buffer, epochs=1, lr=1.0, gamma=1.0,
    )
    assert isinstance(result, dict)
    # terminal successor: both targets equal the reward, gap is zero
    assert result["targets"][0] == 5.0
    assert result["vanilla_targets"][0] == 5.0
    assert result["overestimation_gap"][0] == 0.0
    assert math.isfinite(float(result["estimate"]))


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmddqn as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
