"""Tests for gb_ktv (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb_ktv import gibbons_kendall_tau_var


def test_gb_ktv_basic():
    out = gibbons_kendall_tau_var(10)
    assert out["var_tau"] == pytest.approx(2 * 25 / (9 * 90))


def test_gb_ktv_edge():
    with pytest.raises(ValueError):
        gibbons_kendall_tau_var(1)
