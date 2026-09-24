"""Tests for grmlc.geron_classification_mlp_output."""

from morie.fn import _array_core as np

from morie.fn.grmlc import geron_classification_mlp_output


def test_grmlc_basic():
    """Test basic functionality."""
    a_last = [1.0]
    W_out = [[2.0], [0.0], [-1.0]]
    b_out = [0.0, 0.0, 0.0]
    result = geron_classification_mlp_output(a_last, W_out, b_out)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grmlc_edge():
    """Test edge cases."""
    a_last = [1.0]
    W_out = [[2.0], [0.0], [-1.0]]
    b_out = [0.0, 0.0, 0.0]
    result = geron_classification_mlp_output(a_last, W_out, b_out)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grmlc as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
