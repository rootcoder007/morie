"""Tests for hmfsf.geron_few_shot."""

import doctest as _doctest

import pytest

from morie.fn import _array_core as np

from morie.fn.hmfsf import geron_few_shot


def test_hmfsf_basic():
    """Test basic functionality."""
    def copycat(prompt):
        lines = [l for l in prompt.split("\n") if "->" in l and not l.endswith("-> ")]
        return lines[-1].split("-> ")[1] if lines else "?"

    examples = [("a", "1"), ("b", "2"), ("c", "3"), ("d", "4"), ("e", "5")]
    query = "f"
    k = 2
    result = geron_few_shot(copycat, examples, query, k)
    assert isinstance(result, dict)
    assert result["prediction"] == "2"
    assert result["zero_shot_prediction"] == "?"
    assert result["k"] == 2
    assert result["n_available"] == 5
    assert result["changed_by_context"] is True
    assert "prompt" in result
    assert "zero_shot_prompt" in result
    assert "prompt_length" in result


def test_hmfsf_edge():
    """Test edge cases."""
    def copycat(prompt):
        lines = [l for l in prompt.split("\n") if "->" in l and not l.endswith("-> ")]
        return lines[-1].split("-> ")[1] if lines else "?"

    examples = [("a", "1")]
    query = "c"
    with pytest.raises(ValueError, match="max_context"):
        geron_few_shot(copycat, examples, query, max_context=3)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.



import morie.fn.hmfsf as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
