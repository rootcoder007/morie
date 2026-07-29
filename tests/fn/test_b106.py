"""Tests for b106.burkov_lm_ch1_layer1_output."""

from morie.fn.b106 import burkov_lm_ch1_layer1_output


def test_b106_basic():
    out = burkov_lm_ch1_layer1_output([[1.0], [-1.0]], [2.0], [0.0, 0.0])
    assert out["output"] == [2.0, 0.0]


def test_b106_edge():
    import pytest
    with pytest.raises(ValueError, match="columns"):
        burkov_lm_ch1_layer1_output([[1.0, 2.0]], [1.0], [0.0])
