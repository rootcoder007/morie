"""Tests for km057.kamath_ch4_lora_obj."""

import math

from morie.fn import _array_core as np

from morie.fn.km057 import kamath_ch4_lora_obj


def test_km057_basic():
    """Test basic functionality."""
    # Adapted model (Theta) and base model (Phi_0) as constant-probability functions.
    Theta = lambda xi, y_prefix, y_t: 0.5
    Phi_0 = lambda xi, y_prefix, y_t: 0.25

    # x: list of contexts; y: list of target token sequences.
    x = ["doc1", "doc2"]
    y = [["a", "b"], ["c"]]

    result = kamath_ch4_lora_obj(Theta, Phi_0, x, y)

    # Verify that all documented keys are present in the RichResult.
    for key in ("estimate", "base_objective", "improvement",
                "per_pair", "base_per_pair", "n", "method"):
        assert key in result

    # Numeric outputs must be finite.
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["base_objective"])
    assert math.isfinite(result["improvement"])

    # n equals the number of (x, y) pairs.
    assert result["n"] == len(y)

    # Per-pair lists have the correct length.
    assert len(result["per_pair"]) == len(y)
    assert len(result["base_per_pair"]) == len(y)


def test_km057_edge():
    """Edge case: identical base and adapted models yield zero improvement."""
    # Both models return the same probability for any token.
    Theta = lambda xi, y_prefix, y_t: 0.7
    Phi_0 = lambda xi, y_prefix, y_t: 0.7

    x = ["doc"]
    y = [["a", "b", "c"]]

    result = kamath_ch4_lora_obj(Theta, Phi_0, x, y)

    for key in ("estimate", "base_objective", "improvement",
                "per_pair", "base_per_pair", "n", "method"):
        assert key in result

    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["base_objective"])

    # When the two models are identical, the improvement must be zero (up to floating point).
    assert math.isclose(result["improvement"], 0.0, abs_tol=1e-12)

    assert result["n"] == 1
    assert len(result["per_pair"]) == 1
