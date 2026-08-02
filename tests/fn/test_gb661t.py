"""Tests for gb661t (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb661t import gibbons_mw_ties


def test_gb661t_basic():
    rng = np.random.default_rng(6)
    x = np.round(rng.standard_normal(25), 0); y = np.round(rng.standard_normal(25), 0)
    out = gibbons_mw_ties(x, y)
    assert out["var_corrected"] < out["var_uncorrected"]


def test_gb661t_edge():
    with pytest.raises(ValueError):
        gibbons_mw_ties(np.ones(5), np.ones(5))  # degenerate
