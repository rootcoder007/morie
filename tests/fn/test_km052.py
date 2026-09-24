"""Tests for km052.kamath_ch3_t5_template_obj."""

import math

import pytest
from morie.fn import _array_core as np

from morie.fn.km052 import kamath_ch3_t5_template_obj


def test_km052_basic():
    """Test basic functionality."""
    D_train = [("great movie", "pos"), ("awful movie", "neg"), ("okay film", "neu")]
    T = "{x} It was {y}"
    T5 = lambda T, s: 0.5
    result = kamath_ch3_t5_template_obj(D_train, T, T5)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "per_example" in result
    assert "filled_inputs" in result
    assert "n" in result
    assert "method" in result
    assert result["n"] == 3
    assert math.isfinite(result["estimate"])
    assert result["estimate"] <= 0.0
    assert len(result["per_example"]) == 3
    assert len(result["filled_inputs"]) == 3


def test_km052_edge():
    """Test edge cases."""
    # Empty D_train is explicitly invalid per docstring.
    with pytest.raises(ValueError):
        kamath_ch3_t5_template_obj([], "{x} {y}", lambda T, s: 0.5)
