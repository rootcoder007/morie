"""Tests for alocv.alammar_output_verification."""

from morie.fn.alocv import alammar_output_verification


def test_alocv_basic():
    out = alammar_output_verification("x", ["c1"], lambda r, c: "PASS")
    assert out["passed"] is True


def test_alocv_edge():
    import pytest
    with pytest.raises(ValueError, match="empty gate"):
        alammar_output_verification("x", [], lambda r, c: "PASS")
