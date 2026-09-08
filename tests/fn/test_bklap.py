"""Tests for bklap.burkov_laplace_add_one."""

from morie.fn.bklap import burkov_laplace_add_one


def test_bklap_basic():
    assert burkov_laplace_add_one(0, 0, 4)["estimate"] == 0.25


def test_bklap_edge():
    import pytest
    with pytest.raises(ValueError, match="cannot exceed"):
        burkov_laplace_add_one(5, 4, 3)
