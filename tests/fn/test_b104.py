"""Tests for b104.burkov_lm_ch1_linear_vector."""

from morie.fn.b104 import burkov_lm_ch1_linear_vector


def test_b104_basic():
    assert burkov_lm_ch1_linear_vector([1.0, 2.0], [3.0, 4.0], 0.5)["estimate"] == 11.5


def test_b104_edge():
    import pytest
    with pytest.raises(ValueError, match="same length"):
        burkov_lm_ch1_linear_vector([1.0], [1.0, 2.0], 0.0)
