"""Tests for hmdldqn.geron_dueling_dqn."""

from morie.fn import _array_core as np

from morie.fn.hmdldqn import geron_dueling_dqn


def test_hmdldqn_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    S = 5
    nA = 3
    V = rng.normal(0, 1, S)
    A = rng.normal(0, 1, (S, nA))
    buffer = [
        (0, 0, 1.0, 1, False),
        (1, 1, 0.5, 2, False),
        (2, 0, -0.5, 3, False),
        (3, 2, 1.5, 4, False),
        (4, 1, 0.0, 0, True),
    ]
    epochs = 5
    lr = 0.1
    result = geron_dueling_dqn(None, V, A, buffer, epochs, lr)
    assert isinstance(result, dict)
    assert "Q" in result
    assert "V" in result
    assert "A" in result
    assert "loss_history" in result
    assert "advantage_mean" in result
    assert "value_share" in result
    assert "greedy_policy" in result
    assert "sync_epochs" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result


def test_hmdldqn_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    S = 4
    nA = 2
    V = rng.normal(0, 1, S)
    A = rng.normal(0, 1, (S, nA))
    buffer = [(0, 0, 1.0, 0, True)]
    epochs = 1
    lr = 0.0
    result = geron_dueling_dqn(None, V, A, buffer, epochs, lr)
    assert isinstance(result, dict)
    assert "Q" in result
    assert "V" in result
    assert "A" in result
    assert "loss_history" in result
    assert "advantage_mean" in result


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmdldqn as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
