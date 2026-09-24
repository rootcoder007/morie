"""Tests for grwdc.geron_adamw_decoupled_weight_decay."""

from morie.fn import _array_core as np

from morie.fn.grwdc import geron_adamw_decoupled_weight_decay


def test_grwdc_basic():
    """Test basic functionality."""
    theta = [1.0, 10.0]
    grad = [0.1, 0.1]
    m = [0.0, 0.0]
    s = [0.0, 0.0]
    t = 1
    eta = 0.01
    result = geron_adamw_decoupled_weight_decay(theta, grad, m, s, t, eta)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grwdc_edge():
    """Test edge cases."""
    theta = [1.0, 10.0]
    grad = [0.1, 0.1]
    m = [0.0, 0.0]
    s = [0.0, 0.0]
    t = 1
    eta = 0.01
    result = geron_adamw_decoupled_weight_decay(theta, grad, m, s, t, eta)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grwdc as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
