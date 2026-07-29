"""Tests for b111.burkov_lm_ch1_bce_gradients."""

from morie.fn.b111 import burkov_lm_ch1_bce_gradients


def test_b111_basic():
    out = burkov_lm_ch1_bce_gradients([0.5], [1.0], [[2.0]])
    assert out["grad_w"] == [-1.0]
    assert out["grad_b"] == -0.5


def test_b111_edge():
    import pytest
    with pytest.raises(ValueError, match="dataset size"):
        burkov_lm_ch1_bce_gradients([0.5], [1.0], [[2.0]], N=7)
