"""Tests for hmhfpi.geron_hf_pipelines."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.hmhfpi import geron_hf_pipelines


def test_hmhfpi_basic():
    """Test basic functionality."""
    task = "sentiment-analysis"
    inputs = ["good", "bad"]
    model = lambda xs: [[2.0, 0.0], [0.0, 3.0]]
    result = geron_hf_pipelines(
        task, inputs, model, labels=["POSITIVE", "NEGATIVE"]
    )
    assert isinstance(result, dict)
    assert "predictions" in result
    assert "scores" in result
    assert "raw" in result
    assert "task" in result
    assert result["task"] == task
    # One prediction per input.
    assert len(result["predictions"]) == len(inputs)
    assert len(result["scores"]) == len(inputs)
    # Each prediction is a dict with a label and a probability score.
    for pred in result["predictions"]:
        assert "label" in pred
        assert "score" in pred
        assert math.isfinite(float(pred["score"]))
        assert 0.0 <= float(pred["score"]) <= 1.0
    # Scores are probabilities per input and sum to 1.
    row_sum = float(np.sum(result["scores"][0]))
    assert math.isclose(row_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9)


def test_hmhfpi_edge():
    """Test edge cases."""
    task = "text-classification"
    inputs = ["a", "b"]
    # Model returns one row for two inputs -> documented to raise.
    model = lambda xs: [[1.0, 0.0]]
    with pytest.raises(ValueError):
        geron_hf_pipelines(task, inputs, model)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmhfpi as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
