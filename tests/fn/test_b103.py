"""Tests for b103.burkov_lm_ch1_mse_cost."""

from morie.fn.b103 import burkov_lm_ch1_mse_cost


def test_b103_basic():
    assert burkov_lm_ch1_mse_cost(2.0, 0.0, [1.0, 2.0], [2.0, 4.0])["cost"] == 0.0


def test_b103_edge():
    import pytest
    with pytest.raises(ValueError, match="dataset size"):
        burkov_lm_ch1_mse_cost(1.0, 0.0, [1.0], [1.0], N=9)
