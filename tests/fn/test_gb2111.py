"""Tests for gb2111 (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb2111 import gibbons_tolerance_beta


def test_gb2111_basic():
    from morie.fn import _sci_core as special
    out = gibbons_tolerance_beta(n=50, r=1, s=50, p=0.8)
    assert out["gamma"] == pytest.approx(1 - special.betainc(49, 2, 0.8))


def test_gb2111_edge():
    with pytest.raises(ValueError):
        gibbons_tolerance_beta(n=10, r=5, s=3, p=0.9)
