"""Tests for kmsrag.kamath_self_rag."""

from morie.fn import _array_core as np

from morie.fn.kmsrag import kamath_self_rag


def test_kmsrag_basic():
    """Test basic functionality."""
    context = ["doc1", "doc2"]
    reflection_model = lambda c, q: ["[Retrieve]", "[Relevant]", "[Supported]"]
    result = kamath_self_rag(context, reflection_model)
    assert isinstance(result, dict)
    assert result["retrieve"] is True
    assert result["relevant"] is True
    assert result["supported"] is True
    assert result["estimate"] == 3
    assert result["n"] == 3
    assert "tokens" in result
    assert "by_group" in result
    assert "method" in result


def test_kmsrag_edge():
    """Test edge cases."""
    context = ["doc"]
    reflection_model = lambda c, q: ["[No Retrieve]"]
    result = kamath_self_rag(context, reflection_model)
    assert isinstance(result, dict)
    assert result["retrieve"] is False
    assert result["relevant"] is None
    assert result["supported"] is None
    assert result["estimate"] == 1
    assert result["n"] == 1
    assert result["tokens"] == ["[No Retrieve]"]


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmsrag as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
