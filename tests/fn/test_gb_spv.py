"""Tests for gb_spv (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb_spv import gibbons_spearman_rho_var


def test_gb_spv_basic():
    assert gibbons_spearman_rho_var(10)["var"] == pytest.approx(1 / 9)


def test_gb_spv_edge():
    out = gibbons_spearman_rho_var(26, r_s=0.5)
    assert out["z"] == pytest.approx(2.5)
    with pytest.raises(ValueError):
        gibbons_spearman_rho_var(1)
