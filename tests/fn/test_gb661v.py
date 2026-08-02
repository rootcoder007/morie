"""Tests for gb661v (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb661v import gibbons_mw_var


def test_gb661v_basic():
    assert gibbons_mw_var(6, 8)["var"] == pytest.approx(60.0)


def test_gb661v_edge():
    with pytest.raises(ValueError):
        gibbons_mw_var(0, 5)
