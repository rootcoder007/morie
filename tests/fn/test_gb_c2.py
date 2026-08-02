"""Tests for gb_c2 (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb_c2 import gibbons_chi2_yates


def test_gb_c2_basic():
    from scipy import stats
    tbl = [[18, 7], [6, 19]]
    assert gibbons_chi2_yates(tbl)["chi2_corrected"] == pytest.approx(
        stats.chi2_contingency(tbl, correction=True).statistic, abs=1e-10)


def test_gb_c2_edge():
    with pytest.raises(ValueError):
        gibbons_chi2_yates(np.ones((3, 3)))  # 2x2 only
