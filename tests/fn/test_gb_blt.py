"""Tests for gb_blt (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb_blt import gibbons_balance_incomplete


def test_gb_blt_basic():
    nan = np.nan
    bib = np.array([[1, 2, 3, nan], [1, 2, nan, 3], [1, nan, 2, 3], [nan, 1, 2, 3]])
    out = gibbons_balance_incomplete(bib)
    assert out["lambda_"] == 2 and 0 < out["W_b"] <= 1


def test_gb_blt_edge():
    nan = np.nan
    bad = np.array([[1, 2, 3, nan], [1, 2, 3, nan], [1, nan, 2, 3], [nan, 1, 2, 3]])
    with pytest.raises(ValueError):
        gibbons_balance_incomplete(bad)  # not a BIB design
