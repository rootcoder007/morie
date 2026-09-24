"""Tests for km077.kamath_ch6_factscore."""

from morie.fn import _array_core as np

import math

import pytest

from morie.fn.km077 import kamath_ch6_factscore


def test_km077_basic():
    """Test basic functionality with a small example."""
    # Define a model M that returns the prompt if non-empty, else None.
    M = lambda x: x or None

    # Define a list of prompts.
    X = ["a b", "c", "", "d e"]

    # Define a fact extractor A_y that splits the response into words.
    A_y = lambda r: r.split()

    # Define a knowledge source C as a set of supported atoms.
    C = {"a", "c", "e"}

    # Compute the score.
    result = kamath_ch6_factscore(M, X, A_y, C)

    # Check that the result is a mapping (dict-like).
    assert isinstance(result, dict)

    # Check that all expected keys are present.
    expected_keys = {"estimate", "per_prompt", "n_responded", "response_rate", "n", "method"}
    assert expected_keys.issubset(result.keys())

    # Check that n matches the number of prompts.
    assert result["n"] == len(X)

    # Check that n_responded matches the number of non-empty prompts (M returns None for empty).
    expected_n_responded = sum(1 for x in X if x)
    assert result["n_responded"] == expected_n_responded

    # Check response_rate.
    assert result["response_rate"] == expected_n_responded / len(X)

    # Check that per_prompt is a list of length n_responded.
    assert isinstance(result["per_prompt"], list)
    assert len(result["per_prompt"]) == expected_n_responded
    # Each per_prompt should be a float between 0 and 1.
    for p in result["per_prompt"]:
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    # Check estimate is finite and between 0 and 1.
    assert math.isfinite(result["estimate"])
    assert 0.0 <= result["estimate"] <= 1.0

    # Method string.
    assert result["method"] == "FActScore (Kamath Eq 6.1)"


def test_km077_edge():
    """Test edge case: empty prompts raises ValueError."""
    M = lambda x: x or None
    A_y = lambda r: r.split()
    C = {"a"}
    X = []  # empty list
    with pytest.raises(ValueError):
        kamath_ch6_factscore(M, X, A_y, C)
