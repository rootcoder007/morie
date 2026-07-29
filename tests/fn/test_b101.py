"""Tests for b101.burkov_lm_ch1_linear_function."""

from morie.fn.b101 import burkov_lm_ch1_linear_function


def test_b101_basic():
    out = burkov_lm_ch1_linear_function([1.0, 2.0, 3.0], 2.0, 1.0)
    assert out["predictions"] == [3.0, 5.0, 7.0]


def test_b101_edge():
    out = burkov_lm_ch1_linear_function([-4.0], -0.5, 0.0)
    assert out["predictions"] == [2.0]
