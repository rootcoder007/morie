"""Tests for kmcrb.kamath_cross_encoder_rerank."""

from morie.fn import _array_core as np

from morie.fn.kmcrb import kamath_cross_encoder_rerank


def test_kmcrb_basic():
    """Test basic functionality."""
    q = 'q'
    docs = ['aa', 'b', 'cc']
    model = lambda q, d: len(d)
    result = kamath_cross_encoder_rerank(q, docs, model)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmcrb_edge():
    """Test edge cases."""
    q = 'q'
    docs = ['aa', 'b', 'cc']
    model = lambda q, d: len(d)
    result = kamath_cross_encoder_rerank(q, docs, model)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmcrb as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
