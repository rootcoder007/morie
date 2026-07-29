"""Tests for bknrm.burkov_vector_norm."""

from morie.fn.bknrm import burkov_vector_norm


def test_bknrm_basic():
    assert burkov_vector_norm([3.0, 4.0])["estimate"] == 5.0


def test_bknrm_edge():
    assert burkov_vector_norm([0.0, 0.0])["estimate"] == 0.0
