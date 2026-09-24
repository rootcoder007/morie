"""Tests for kmcrag.kamath_corrective_rag."""

from morie.fn import _array_core as np

from morie.fn.kmcrag import kamath_corrective_rag


def test_kmcrag_basic():
    """Test basic functionality."""
    query = 'q'
    docs = ['d1', 'd2']
    clf = lambda q, d: 0.9 if d == 'd1' else 0.1
    tau_hi = 0.8
    tau_lo = 0.2
    result = kamath_corrective_rag(query, docs, clf, tau_hi, tau_lo)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmcrag_edge():
    """Test edge cases."""
    query = 'q'
    docs = ['d1', 'd2']
    clf = lambda q, d: 0.9 if d == 'd1' else 0.1
    tau_hi = 0.8
    tau_lo = 0.2
    result = kamath_corrective_rag(query, docs, clf, tau_hi, tau_lo)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmcrag as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
