"""Tests for kmap.kamath_autoprompt_gradient_search."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.kmap import kamath_autoprompt_gradient_search


def test_kmap_basic():
    """Test basic functionality: greedy loss-minimising trigger search."""
    template = ["x", None]
    dataset = [1]
    vocab = ["a", "b"]
    # Loss is 1.0 if the slot is 'a', else 0.5; the search must pick 'b'.
    model = lambda tpl, d: 1.0 if tpl[1] == "a" else 0.5
    result = kamath_autoprompt_gradient_search(
        template, dataset, model, vocab=vocab)

    # RichResult is dict-like: the docstring indexes it by key.
    assert isinstance(result, dict)
    for key in ("estimate", "loss", "trigger_tokens", "prompt",
                "positions", "history", "n", "method"):
        assert key in result

    # Documented contract: one trigger slot is filled with 'b', final loss 0.5.
    assert result["trigger_tokens"] == ["b"]
    assert result["estimate"] == 0.5
    assert math.isfinite(result["estimate"])
    assert result["estimate"] == result["loss"]
    assert result["n"] == 1
    assert result["positions"] == [1]
    assert result["prompt"] == ["x", "b"]


def test_kmap_edge():
    """Test edge cases: documented invalid inputs raise ValueError."""
    dataset = [1]
    vocab = ["a", "b"]

    # Template contains no None trigger slots.
    with pytest.raises(ValueError):
        kamath_autoprompt_gradient_search(
            ["x", "y"], dataset, lambda t, d: 0.0, vocab=vocab)

    # vocab must be non-empty.
    with pytest.raises(ValueError):
        kamath_autoprompt_gradient_search(
            [None, "y"], dataset, lambda t, d: 0.0, vocab=[])

    # model must be callable.
    with pytest.raises(ValueError):
        kamath_autoprompt_gradient_search(
            [None, "y"], dataset, "not_callable", vocab=vocab)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmap as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
