"""Tests for kmstb.kamath_step_back_prompting."""

from morie.fn import _array_core as np

from morie.fn.kmstb import kamath_step_back_prompting


def test_kmstb_basic():
    """Test basic functionality."""
    docs = {"physics": ["d1", "d2"], "which force": ["d2", "d3"]}
    result = kamath_step_back_prompting(
        "which force", lambda q: "physics",
        retrieve=lambda q: docs[q],
        answer=lambda q, ctx: "gravity, per " + ",".join(ctx))
    assert isinstance(result, dict)
    assert result["step_back_query"] == "physics"
    assert result["query"] == "which force"
    assert result["context"] == ["d1", "d2", "d3"]
    assert result["answer"] == "gravity, per d1,d2,d3"
    assert result["stepped_back"] is True
    assert result["n_context"] == 3
    assert result["retrieved_by_query"]["physics"] == ["d1", "d2"]
    assert result["retrieved_by_query"]["which force"] == ["d2", "d3"]


def test_kmstb_edge():
    """Test edge case: model returns the original query, no retrieve/answer."""
    result = kamath_step_back_prompting("what is x", lambda q: "what is x")
    assert isinstance(result, dict)
    assert result["stepped_back"] is False
    assert result["n_context"] == 0
    assert result["context"] == []
    assert "warning" in result
    assert "answer" not in result


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmstb as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
