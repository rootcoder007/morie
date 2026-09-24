"""Tests for hmqat.geron_quantization_aware_training."""

from morie.fn import _array_core as np

from morie.fn.hmqat import geron_quantization_aware_training


def test_hmqat_basic():
    """Test basic functionality."""
    model = 0.5
    X = 0.5
    y = 0.5
    result = geron_quantization_aware_training(model, X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "weights" in result


def test_hmqat_edge():
    """Test edge cases."""
    model = 0.5
    X = 0.5
    y = 0.5
    result = geron_quantization_aware_training(model, X, y)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmqat as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
