"""Tests for km132.kamath_ch9_llm_signal_tokens."""

import pytest
from morie.fn import _array_core as np

from morie.fn.km132 import kamath_ch9_llm_signal_tokens


def test_km132_basic():
    """Test basic functionality."""
    P_X = np.array([[0.0]])
    F_T = np.array([[1.0]])

    def llm(p, f):
        return ("a cat", ["<IMG>", "<AUDIO>"])

    result = kamath_ch9_llm_signal_tokens(P_X, F_T, llm=llm)
    assert isinstance(result, dict)
    assert result["text"] == "a cat"
    assert result["signal_tokens"] == ["<IMG>", "<AUDIO>"]
    assert result["estimate"] == 2
    assert result["n"] == 2
    assert result["generates_modality"] is True
    assert result["method"] == "LLM text and signal tokens (Kamath Eq 9.4)"


def test_km132_edge():
    """Test edge cases."""
    P_X = np.array([[0.0]])
    F_T = np.array([[1.0]])

    # Edge: empty signal tokens (valid input)
    def llm_empty(p, f):
        return ("hello", [])

    result = kamath_ch9_llm_signal_tokens(P_X, F_T, llm=llm_empty)
    assert isinstance(result, dict)
    assert result["estimate"] == 0
    assert result["n"] == 0
    assert result["generates_modality"] is False
    assert result["text"] == "hello"
    assert result["signal_tokens"] == []

    # Edge: llm not callable is invalid per docstring
    with pytest.raises(ValueError):
        kamath_ch9_llm_signal_tokens(P_X, F_T, llm=None)
