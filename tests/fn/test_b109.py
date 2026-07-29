"""Tests for b109.burkov_lm_ch1_binary_cross_entropy."""

from morie.fn.b109 import burkov_lm_ch1_binary_cross_entropy


def test_b109_basic():
    assert burkov_lm_ch1_binary_cross_entropy(1.0, 1.0)["estimate"] == 0.0


def test_b109_edge():
    import pytest
    with pytest.raises(ValueError, match="targets"):
        burkov_lm_ch1_binary_cross_entropy(0.5, 0.3)
