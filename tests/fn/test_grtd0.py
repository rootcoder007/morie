"""Tests for grtd0.geron_td_zero_update."""

from morie.fn import _array_core as np

from morie.fn.grtd0 import geron_td_zero_update


def test_grtd0_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    V = rng.normal(0, 1, 5)
    state = 0
    next_state = 1
    reward = 1.0
    alpha = 0.05
    gamma = 0.9
    result = geron_td_zero_update(V, state, next_state, reward, alpha, gamma)
    assert isinstance(result, dict)
    assert "td_error" in result
    assert "new_value" in result
    assert "target" in result
    assert "V" in result
    assert len(result["V"]) == 5
    import math
    assert math.isfinite(result["td_error"])
    assert math.isfinite(result["new_value"])


def test_grtd0_edge():
    """Test edge case: terminal transition (done=True)."""
    rng = np.random.default_rng(42)
    V = rng.normal(0, 1, 10)
    state = 3
    next_state = 7
    reward = 2.5
    alpha = 0.5
    gamma = 0.0
    result = geron_td_zero_update(V, state, next_state, reward, alpha, gamma, done=True)
    assert isinstance(result, dict)
    assert "td_error" in result
    assert "target" in result
    assert "V" in result
    assert len(result["V"]) == 10
    import math
    assert math.isfinite(result["td_error"])
    # With done=True the target equals the reward exactly
    assert result["target"] == reward


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grtd0 as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
