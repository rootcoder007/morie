"""Tests for kmcai.kamath_constitutional_ai_loop."""

from morie.fn.kmcai import kamath_constitutional_ai_loop


def test_kmcai_basic():
    """Test basic functionality."""
    initial_response = "no"
    constitution = ["be kind", "be brief"]

    def model(stage, principle, response, critique):
        return "too blunt" if stage == "critique" else response + "!"

    result = kamath_constitutional_ai_loop(initial_response, constitution, model)
    assert isinstance(result, dict)
    assert result["revised_response"] == "no!!"
    assert result["n_revisions"] == 2
    assert result["initial_response"] == "no"
    assert "history" in result
    assert isinstance(result["history"], list)
    assert len(result["history"]) == 2
    for entry in result["history"]:
        assert "principle" in entry
        assert "critique" in entry
        assert "response_before" in entry
        assert "response_after" in entry


def test_kmcai_edge():
    """Test edge cases."""
    # single-principle constitution: revisions still compose over one step
    initial_response = "hi"
    constitution = ["be polite"]

    def model(stage, principle, response, critique):
        return "rude" if stage == "critique" else response + "."

    result = kamath_constitutional_ai_loop(initial_response, constitution, model)
    assert isinstance(result, dict)
    assert result["revised_response"] == "hi."
    assert result["n_revisions"] == 1
    assert len(result["history"]) == 1
    assert result["history"][0]["principle"] == "be polite"


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmcai as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
