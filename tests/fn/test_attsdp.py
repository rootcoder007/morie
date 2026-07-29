"""Tests for attsdp.scaled_dot_product_attention."""

from morie.fn.attsdp import scaled_dot_product_attention


def test_attsdp_basic():
    out = scaled_dot_product_attention([[1.0, 0.0]],
        [[1.0, 0.0], [0.0, 1.0]], [[1.0], [0.0]])
    assert abs(sum(out["attention"][0]) - 1.0) < 1e-12


def test_attsdp_edge():
    import pytest
    with pytest.raises(ValueError, match="share d_k"):
        scaled_dot_product_attention([[1.0]], [[1.0, 2.0]], [[1.0]])
