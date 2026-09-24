"""Tests for kmrlaif.kamath_rlaif_objective."""

from morie.fn import _array_core as np

import math

import pytest

from morie.fn.kmrlaif import kamath_rlaif_objective


def test_kmrlaif_basic():
    """Test basic functionality."""
    # A minimal strongly connected preference set: two items, each beats the other.
    ai_preferences = [(0, 1), (1, 0)]
    result = kamath_rlaif_objective(ai_preferences)
    # The result is a dict-like RichResult
    assert isinstance(result, dict)
    # The documented payload keys are present
    expected_keys = {
        "items", "strengths", "scores", "loss", "accuracy",
        "n_preferences", "estimate",
    }
    assert expected_keys.issubset(result.keys())
    # The number of preferences matches the input
    assert result["n_preferences"] == len(ai_preferences)
    # Strengths are a probability distribution (sum to 1)
    assert math.isclose(sum(result["strengths"]), 1.0, rel_tol=1e-9, abs_tol=1e-12)
    # The training loss (estimate) is finite
    assert math.isfinite(result["estimate"])
    # Accuracy is finite
    assert math.isfinite(result["accuracy"])
    # The items list reflects the unique integers used (sorted)
    assert result["items"] == [0, 1]


def test_kmrlaif_edge():
    """Test edge cases: invalid inputs raise ValueError."""
    # Empty list of preferences
    with pytest.raises(ValueError):
        kamath_rlaif_objective([])
    # A pair where the same item is both winner and loser
    with pytest.raises(ValueError):
        kamath_rlaif_objective([(0, 0)])
    # Not strongly connected (only one direction)
    with pytest.raises(ValueError):
        kamath_rlaif_objective([(0, 1)])


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.


def test_every_printed_value_in_the_worked_example_reproduces():
    # Verify the examples from the module's docstring without using the
    # doctest module (which is not in the allowed import list).
    # Example 1: two-item symmetric preferences yield equal strengths.
    out = kamath_rlaif_objective([(0, 1), (1, 0)])
    assert [round(v, 9) for v in out["strengths"]] == [0.5, 0.5]

    # Example 2: three preferences (two beats of 0 over 1, one beat of 1 over 0)
    # give strengths of 2/3 and 1/3.
    out2 = kamath_rlaif_objective([(0, 1), (0, 1), (1, 0)])
    assert abs(out2["strengths"][0] - 2 / 3) < 1e-9
    # The log-strength difference equals log(2) within tolerance.
    assert abs(out2["scores"][0] - out2["scores"][1] - math.log(2)) < 1e-9
