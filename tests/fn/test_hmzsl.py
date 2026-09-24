"""Tests for hmzsl.geron_zero_shot."""

import math

from morie.fn import _array_core as np

from morie.fn.hmzsl import geron_zero_shot


def test_hmzsl_basic():
    """Test basic functionality with a dict-returning model."""
    scores = {"negative": 0.0, "positive": 1.0}
    model = lambda p: scores
    prompt = "Review: it was great. Sentiment:"
    result = geron_zero_shot(model, prompt)
    assert isinstance(result, dict)
    expected_keys = {
        "probabilities", "predicted", "predicted_label", "margin",
        "entropy", "calibrated", "estimate", "n", "method",
    }
    assert expected_keys.issubset(result.keys())
    assert result["predicted_label"] == "positive"
    probs = [float(v) for v in result["probabilities"]]
    assert len(probs) == 2
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert abs(sum(probs) - 1.0) < 1e-6
    assert math.isfinite(float(result["margin"]))
    assert math.isfinite(float(result["entropy"]))
    assert int(result["n"]) == 2


def test_hmzsl_edge():
    """Test calibration with null_prompt yields uniform probabilities."""
    f = lambda p: ([5.0, 0.0] if p == "" else [6.0, 1.0])
    result = geron_zero_shot(f, "x", labels=["a", "b"], null_prompt="")
    assert isinstance(result, dict)
    assert result["calibrated"]
    probs = [float(v) for v in result["probabilities"]]
    assert len(probs) == 2
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert abs(probs[0] - probs[1]) < 1e-9
    assert abs(sum(probs) - 1.0) < 1e-6


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmzsl as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
