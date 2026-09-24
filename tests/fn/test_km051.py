"""Tests for km051.kamath_ch3_qa_trigger_template."""

from morie.fn import _array_core as np

import pytest

from morie.fn.km051 import kamath_ch3_qa_trigger_template


def test_km051_basic():
    """Test basic functionality with default n_triggers."""
    x = "Where?"
    y = "Paris."
    T = "the"
    z_adv = "Rome"
    result = kamath_ch3_qa_trigger_template(x, y, T, z_adv)
    assert isinstance(result, dict)
    # n_triggers defaults to 3
    assert result["n_triggers"] == 3
    expected_prompt = f"Question: {x} Context: {y} Answer: {T} {T} {T} {z_adv}"
    assert result["prompt"] == expected_prompt
    assert result["trigger"] == T
    assert result["adversarial_answer"] == z_adv
    assert isinstance(result["tokens"], list)
    assert result["n"] == len(result["tokens"])
    assert result["estimate"] == float(len(result["tokens"]))
    assert result["method"] == "adversarial-trigger QA prompt (Kamath Eq 3.10)"


def test_km051_edge():
    """Test edge case with n_triggers=1."""
    x = "What is the capital of France?"
    y = "It is Paris."
    T = "alpha"
    z_adv = "beta"
    result = kamath_ch3_qa_trigger_template(x, y, T, z_adv, n_triggers=1)
    assert isinstance(result, dict)
    assert result["n_triggers"] == 1
    expected_prompt = f"Question: {x} Context: {y} Answer: {T} {z_adv}"
    assert result["prompt"] == expected_prompt
    assert result["trigger"] == T
    assert result["adversarial_answer"] == z_adv
    assert isinstance(result["tokens"], list)
    assert result["n"] == len(result["tokens"])
    assert result["estimate"] == float(result["n"])
