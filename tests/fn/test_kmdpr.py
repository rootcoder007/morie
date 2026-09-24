"""Tests for kmdpr.kamath_dense_passage_retrieval."""

from morie.fn import _array_core as np

from morie.fn.kmdpr import kamath_dense_passage_retrieval


def test_kmdpr_basic():
    """Test basic functionality."""
    q_embed = [1.0, 0.0]
    p_embeds = [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]]
    k = 2
    result = kamath_dense_passage_retrieval(q_embed, p_embeds, k)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmdpr_edge():
    """Test edge cases."""
    q_embed = [1.0, 0.0]
    p_embeds = [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]]
    k = 2
    result = kamath_dense_passage_retrieval(q_embed, p_embeds, k)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmdpr as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
