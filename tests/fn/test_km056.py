"""Tests for km056.kamath_ch4_full_finetune_obj."""

from morie.fn import _array_core as np
import math
import pytest

from morie.fn.km056 import kamath_ch4_full_finetune_obj


def test_km056_basic():
    """Test basic functionality."""
    # Define a simple model that returns a constant probability
    Phi = lambda xi, pre, t: 0.5

    # Define contexts and target sequences
    x = ["doc1", "doc2"]
    y = [["a", "b"], ["c", "d", "e"]]

    result = kamath_ch4_full_finetune_obj(Phi, x, y)

    # The result is a RichResult (dict-like)
    assert isinstance(result, dict)

    # Check that all expected keys are present
    expected_keys = {"estimate", "per_pair", "n_tokens", "n", "method"}
    assert set(result.keys()) == expected_keys

    # Compute expected estimate: total tokens * log(0.5)
    total_tokens = sum(len(seq) for seq in y)
    expected_estimate = total_tokens * math.log(0.5)

    # Check estimate is finite and matches expected value within tolerance
    assert math.isfinite(result["estimate"])
    assert abs(result["estimate"] - expected_estimate) < 1e-12

    # Check n_tokens
    assert result["n_tokens"] == total_tokens

    # Check n (number of pairs)
    assert result["n"] == len(y)

    # Check method is a string
    assert isinstance(result["method"], str)


def test_km056_edge():
    """Test edge case with minimal input."""
    # Single context and single token sequence
    Phi = lambda xi, pre, t: 0.5
    x = ["ctx"]
    y = [["token"]]

    result = kamath_ch4_full_finetune_obj(Phi, x, y)

    assert isinstance(result, dict)

    expected_keys = {"estimate", "per_pair", "n_tokens", "n", "method"}
    assert set(result.keys()) == expected_keys

    # Expected estimate: 1 * log(0.5)
    expected_estimate = math.log(0.5)
    assert math.isfinite(result["estimate"])
    assert abs(result["estimate"] - expected_estimate) < 1e-12

    assert result["n_tokens"] == 1
    assert result["n"] == 1
    assert isinstance(result["method"], str)
