"""Tests for gb_cc (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb_cc import gibbons_continuity_corr


def test_gb_cc_basic():
    out = gibbons_continuity_corr(60, 50, 5.0)
    assert out["z_corrected"] == pytest.approx(1.9)
    assert out["p_two_sided"] > out["p_uncorrected"]  # always weakens


def test_gb_cc_edge():
    with pytest.raises(ValueError):
        gibbons_continuity_corr(60, 50, 0.0)
