"""Tests for km144.kamath_ch9_mm_instr_predict."""

import math

from morie.fn import _array_core as np

from morie.fn.km144 import kamath_ch9_mm_instr_predict


def test_km144_basic():
    """Test basic functionality with a callable model."""
    I = "What animal is in this picture?"
    M = "<image data>"
    theta = lambda i, m: "a cat"
    result = kamath_ch9_mm_instr_predict(I, M, theta)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "answer" in result
    assert result["answer"] == "a cat"
    assert result["instruction"] == I
    assert result["multimodal_input"] == M
    assert result["n"] == 1


def test_km144_edge():
    """Test edge case: callable f with theta as parameters."""
    def f(i, m, theta):
        return theta["prefix"] + i

    I = " an animal?"
    M = "<image>"
    theta = {"prefix": "It is"}
    result = kamath_ch9_mm_instr_predict(I, M, theta, f=f)
    assert isinstance(result, dict)
    assert result["answer"] == "It is an animal?"
    assert result["estimate"] == "It is an animal?"
    assert math.isfinite(result["n"])
