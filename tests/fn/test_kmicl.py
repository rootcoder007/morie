"""Tests for kmicl.kamath_in_context_learning_prob."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.kmicl import kamath_in_context_learning_prob


def test_kmicl_basic():
    """Test basic functionality."""
    demonstrations = ["1 -> odd", "2 -> even", "4 -> even"]
    query = "3 ->"
    def model(prompt, answer):
        # deterministic model: returns 0.5 if answer is "odd", else 0.1
        return 0.5 if answer == "odd" else 0.1
    result = kamath_in_context_learning_prob(demonstrations, query, model, answer="odd")
    # result is a RichResult (dict-like)
    assert isinstance(result, dict)
    # Check that all documented keys are present
    for key in ("estimate", "probability", "log_prob", "prompt", "K", "n", "method"):
        assert key in result
    # estimate matches model output
    assert result["estimate"] == 0.5
    # probability is the same as estimate
    assert result["probability"] == 0.5
    # K equals number of demonstrations
    assert result["K"] == len(demonstrations)
    # n equals number of parts (demos + query)
    assert result["n"] == len(demonstrations) + 1
    # prompt contains all demonstrations and the query
    prompt = result["prompt"]
    for d in demonstrations:
        assert d in prompt
    assert query in prompt
    # log_prob is finite and equals log(estimate) when estimate > 0
    assert math.isfinite(result["log_prob"])
    assert math.isclose(result["log_prob"], math.log(0.5))


def test_kmicl_edge():
    """Test edge cases."""
    # Non-callable model should raise ValueError
    with pytest.raises(ValueError):
        kamath_in_context_learning_prob(["d"], "q", model=None)
    # Model returning probability > 1 should raise ValueError
    def bad_model(prompt, answer):
        return 1.5
    with pytest.raises(ValueError):
        kamath_in_context_learning_prob(["d"], "q", model=bad_model)
    # Model returning negative probability should raise ValueError
    def negative_model(prompt, answer):
        return -0.1
    with pytest.raises(ValueError):
        kamath_in_context_learning_prob(["d"], "q", model=negative_model)
    # Model returning non-numeric should raise ValueError
    def non_numeric_model(prompt, answer):
        return "abc"
    with pytest.raises(ValueError):
        kamath_in_context_learning_prob(["d"], "q", model=non_numeric_model)
    # Edge: empty demonstration list
    def zero_model(prompt, answer):
        return 0.0
    result = kamath_in_context_learning_prob([], "q", zero_model)
    assert result["K"] == 0
    assert result["estimate"] == 0.0
    assert result["log_prob"] == -math.inf
    # Edge: custom separator
    def model2(prompt, answer):
        return 0.3
    result = kamath_in_context_learning_prob(["a", "b"], "c", model2, sep="|")
    assert result["prompt"] == "a|b|c"
    assert result["K"] == 2


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmicl as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
