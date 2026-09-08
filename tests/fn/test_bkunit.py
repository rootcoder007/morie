"""Tests for bkunit.burkov_unit_vector."""

from morie.fn.bkunit import burkov_unit_vector


def test_bkunit_basic():
    assert burkov_unit_vector([3.0, 4.0])["unit"] == [0.6, 0.8]


def test_bkunit_edge():
    import pytest
    with pytest.raises(ValueError, match="zero vector"):
        burkov_unit_vector([0.0, 0.0])
