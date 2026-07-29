"""Tests for b107.burkov_lm_ch1_layer2_output."""

from morie.fn.b107 import burkov_lm_ch1_layer2_output


def test_b107_basic():
    assert burkov_lm_ch1_layer2_output([1.0, -1.0], [3.0, 1.0], 0.5)["estimate"] == 2.5


def test_b107_edge():
    import pytest
    with pytest.raises(ValueError, match="weights"):
        burkov_lm_ch1_layer2_output([1.0], [1.0, 2.0], 0.0)
