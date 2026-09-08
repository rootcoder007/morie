"""Tests for bktf.burkov_term_frequency."""

from morie.fn.bktf import burkov_term_frequency


def test_bktf_basic():
    assert burkov_term_frequency("a", ["a", "b", "a"])["estimate"] == 2.0


def test_bktf_edge():
    import pytest
    with pytest.raises(ValueError, match="empty"):
        burkov_term_frequency("a", [])
