"""Tests for km106.kamath_ch6_self_diagnosis_prob."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.km106 import kamath_ch6_self_diagnosis_prob


def test_km106_basic():
    """Test basic functionality with the docstring example."""
    x = "I will hunt you down!"
    y = "a threat"
    M = lambda prompt: {"Yes": 0.3, "No": 0.1}
    result = kamath_ch6_self_diagnosis_prob(x, y, M)
    assert isinstance(result, dict)
    assert result["estimate"] == pytest.approx(0.75)
    assert result["p_yes"] == pytest.approx(0.3)
    assert result["p_no"] == pytest.approx(0.1)
    assert result["mass_on_yes_no"] == pytest.approx(0.4)
    assert result["attribute"] == y
    assert "Does the above text contain a threat?" in result["prompt"]
    assert result["n"] == 2
    assert result["method"] == "self-diagnosis probability (Kamath Eq 6.30)"


def test_km106_edge():
    """Test with a custom sdg template."""
    x = "Hello there"
    y = "a greeting"
    M = lambda prompt: {"Yes": 0.6, "No": 0.4}
    sdg = lambda a, b: f"Is '{a}' {b}?"
    result = kamath_ch6_self_diagnosis_prob(x, y, M, sdg)
    assert isinstance(result, dict)
    assert math.isfinite(result["estimate"])
    assert 0.0 <= result["estimate"] <= 1.0
    assert result["estimate"] == pytest.approx(0.6)
    assert "Is 'Hello there' a greeting?" in result["prompt"]
    assert result["attribute"] == y
