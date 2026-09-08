"""Tests for attmh.multi_head_attention."""

from morie.fn.attmh import multi_head_attention


def test_attmh_basic():
    I2 = [[1.0, 0.0], [0.0, 1.0]]
    out = multi_head_attention(I2, I2, I2, [I2], [I2], [I2], I2, 1)
    assert len(out["output"]) == 2


def test_attmh_edge():
    import pytest
    I2 = [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ValueError, match="one per head"):
        multi_head_attention(I2, I2, I2, [I2], [I2], [I2], I2, 2)
