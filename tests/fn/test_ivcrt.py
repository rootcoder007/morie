"""Tests for ivcrt.iv_conditions."""

import pytest

from morie.fn.ivcrt import iv_conditions


def test_ivcrt_basic():
    valid = {"Z": ["X"], "U": ["X", "Y"], "X": ["Y"]}
    out = iv_conditions(valid, "Z", "X", "Y")
    assert out["valid"] is True
    direct = {"Z": ["X", "Y"], "U": ["X", "Y"], "X": ["Y"]}
    assert iv_conditions(direct, "Z", "X", "Y")["exclusion_independence"] is False


def test_ivcrt_edge():
    with pytest.raises(ValueError):
        iv_conditions({"Z": ["X"], "X": ["Y"]}, "W", "X", "Y")  # W absent
