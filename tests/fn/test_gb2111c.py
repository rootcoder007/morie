"""Tests for gb2111c (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb2111c import gibbons_elementary_coverage_beta


def test_gb2111c_basic():
    out = gibbons_elementary_coverage_beta(20)
    assert out["mean"] == pytest.approx(1 / 21)


def test_gb2111c_edge():
    with pytest.raises(ValueError):
        gibbons_elementary_coverage_beta(0)
