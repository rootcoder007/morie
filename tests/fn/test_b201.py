"""Tests for b201.burkov_lm_ch2_categorical_cross_entropy."""

from morie.fn.b201 import burkov_lm_ch2_categorical_cross_entropy


def test_b201_basic():
    import math
    out = burkov_lm_ch2_categorical_cross_entropy([0.25, 0.75], 1)
    assert out["estimate"] == -math.log(0.75)


def test_b201_edge():
    import pytest
    with pytest.raises(ValueError, match="probability distribution"):
        burkov_lm_ch2_categorical_cross_entropy([2.0, 1.0], 0)
