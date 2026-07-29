"""Tests for bkbpc.burkov_bits_per_character."""

from morie.fn.bkbpc import burkov_bits_per_character


def test_bkbpc_basic():
    import math
    out = burkov_bits_per_character(math.log(2), 50, 50)
    assert abs(out["estimate"] - 1.0) < 1e-12


def test_bkbpc_edge():
    import pytest
    with pytest.raises(ValueError, match="positive"):
        burkov_bits_per_character(1.0, 0, 10)
