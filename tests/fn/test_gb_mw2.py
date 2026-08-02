"""Tests for gb_mw2 (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb_mw2 import gibbons_mw_rs_equiv


def test_gb_mw2_basic():
    rng = np.random.default_rng(9)
    out = gibbons_mw_rs_equiv(rng.standard_normal(10), rng.standard_normal(12))
    assert out["identity_holds"] is True


def test_gb_mw2_edge():
    with pytest.raises(ValueError):
        gibbons_mw_rs_equiv([], [1.0])
