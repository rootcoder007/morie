"""Tests for km092.kamath_ch6_stereotypical_assoc."""

from morie.fn import _array_core as np

from morie.fn.km092 import kamath_ch6_stereotypical_assoc


def test_km092_basic():
    """Test basic functionality."""
    w = "nurse"
    A_i = ["she"]
    Yhat = ["she is a nurse", "she is a doctor"]
    result = kamath_ch6_stereotypical_assoc(w, A_i, Yhat)
    assert isinstance(result, dict)
    assert result["estimate"] == 1.0
    assert result["word"] == "nurse"
    assert result["per_attribute"] == {"she": 1}
    assert result["n_outputs_with_w"] == 1
    assert result["n"] == 2


def test_km092_edge():
    """Test edge case: w mentioned multiple times in the same output."""
    w = "nurse"
    A_i = ["she"]
    Yhat = ["she and she are nurse", "she is a doctor"]
    result = kamath_ch6_stereotypical_assoc(w, A_i, Yhat)
    assert isinstance(result, dict)
    assert result["estimate"] == 2.0
    assert result["word"] == "nurse"
    assert result["per_attribute"] == {"she": 2}
    assert result["n_outputs_with_w"] == 1
    assert result["n"] == 2
