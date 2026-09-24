"""Tests for kmfew.kamath_few_shot_exemplar_selection."""

from morie.fn import _array_core as np

from morie.fn.kmfew import kamath_few_shot_exemplar_selection


def test_kmfew_basic():
    """Test basic functionality."""
    D = 0.5
    query_embed = 0.5
    K = 1
    result = kamath_few_shot_exemplar_selection(D, query_embed, K)
    assert isinstance(result, dict)
    assert "estimate" in result or "selected" in result


def test_kmfew_edge():
    """Test edge cases."""
    D = 0.5
    query_embed = 0.5
    K = 1
    result = kamath_few_shot_exemplar_selection(D, query_embed, K)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmfew as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
