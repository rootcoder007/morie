"""Tests for bkdot.burkov_dot_product."""

from morie.fn.bkdot import burkov_dot_product


def test_bkdot_basic():
    assert burkov_dot_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])["estimate"] == 32.0


def test_bkdot_edge():
    import pytest
    with pytest.raises(ValueError, match="same length"):
        burkov_dot_product([1.0], [1.0, 2.0])
