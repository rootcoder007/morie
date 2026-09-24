"""Tests for kmarel.kamath_ragas_answer_relevance."""

import math

from morie.fn import _array_core as np

from morie.fn.kmarel import kamath_ragas_answer_relevance


def test_kmarel_basic():
    """Test basic functionality."""
    answer = "The cat sat on the mat."
    original_question = [1.0, 0.0]
    # model(answer) returns the reverse-generated question EMBEDDINGS (n x d).
    model = lambda a: [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    result = kamath_ragas_answer_relevance(answer, original_question, model)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "score" in result
    assert "similarities" in result
    assert "n" in result
    assert "method" in result
    assert result["n"] == 3
    assert math.isfinite(result["estimate"])
    assert -1.0 <= result["estimate"] <= 1.0
    assert len(result["similarities"]) == 3


def test_kmarel_edge():
    """Test edge cases: model returns text and embed callable is supplied."""
    answer = "Paris is the capital of France."
    original_question = "What is the capital of France?"
    # model returns text reverse-questions; embed turns each into a vector.
    model = lambda a: ["reverse_q_1", "reverse_q_2"]
    embed = lambda q: [1.0, 0.0] if q == "reverse_q_1" else [0.0, 1.0]
    result = kamath_ragas_answer_relevance(
        answer, original_question, model, embed=embed
    )
    assert isinstance(result, dict)
    assert "estimate" in result
    assert result["n"] == 2
    assert math.isfinite(result["estimate"])
    assert -1.0 <= result["estimate"] <= 1.0


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmarel as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
