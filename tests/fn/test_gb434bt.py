"""Tests for gb434bt (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb434bt import gibbons_ks_bt_formula


def test_gb434bt_basic():
    from scipy import stats
    assert gibbons_ks_bt_formula(0.25, 20)["p_exceed"] == pytest.approx(
        stats.ksone.sf(0.25, 20), abs=1e-10)


def test_gb434bt_edge():
    with pytest.raises(ValueError):
        gibbons_ks_bt_formula(1.5, 10)
