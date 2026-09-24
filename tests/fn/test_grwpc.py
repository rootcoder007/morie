"""Tests for grwpc.geron_wordpiece_tokenizer_score."""

from morie.fn import _array_core as np

from morie.fn.grwpc import geron_wordpiece_tokenizer_score


def test_grwpc_basic():
    """Test basic functionality."""
    counts = {'h': 15, 'u': 20, 'g': 4, 's': 5}
    pairs = {('h', 'u'): 10, ('g', 's'): 4}
    result = geron_wordpiece_tokenizer_score(counts, pairs)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grwpc_edge():
    """Test edge cases."""
    counts = {'h': 15, 'u': 20, 'g': 4, 's': 5}
    pairs = {('h', 'u'): 10, ('g', 's'): 4}
    result = geron_wordpiece_tokenizer_score(counts, pairs)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grwpc as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
