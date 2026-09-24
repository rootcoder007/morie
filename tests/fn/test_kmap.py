"""Tests for kmap.kamath_autoprompt_gradient_search."""

from morie.fn import _array_core as np

from morie.fn.kmap import kamath_autoprompt_gradient_search


def test_kmap_basic():
    """Test basic functionality."""
    template = [None, 'y']
    dataset = [1]
    model = lambda tpl, d: 0.0
    result = kamath_autoprompt_gradient_search(template, dataset, model)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmap_edge():
    """Test edge cases."""
    template = [None, 'y']
    dataset = [1]
    model = lambda tpl, d: 0.0
    result = kamath_autoprompt_gradient_search(template, dataset, model)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmap as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
