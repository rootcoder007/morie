"""Tests for gb_wcin (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb_wcin import gibbons_concordance_incomplete


def test_gb_wcin_basic():
    nan = np.nan
    agree = np.array([[1, 2, 3, nan], [nan, 1, 2, 3], [1, 2, nan, 3]])
    disagree = np.array([[3, 2, 1, nan], [nan, 3, 2, 1], [1, 2, nan, 3]])
    assert (gibbons_concordance_incomplete(agree)["W"]
            > gibbons_concordance_incomplete(disagree)["W"])


def test_gb_wcin_edge():
    nan = np.nan
    with pytest.raises(ValueError):
        gibbons_concordance_incomplete(np.array([[1, nan], [2, nan]]))
