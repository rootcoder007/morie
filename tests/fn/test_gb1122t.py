"""Tests for gb1122t (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb1122t import gibbons_kendall_ties


def test_gb1122t_basic():
    from scipy import stats
    x = np.array([1, 2, 2, 3, 4, 4], float); y = np.array([2, 1, 3, 3, 5, 4], float)
    assert gibbons_kendall_ties(x, y)["tau_b"] == pytest.approx(
        stats.kendalltau(x, y).statistic, abs=1e-12)


def test_gb1122t_edge():
    with pytest.raises(ValueError):
        gibbons_kendall_ties([1, 1], [1, 1])  # all tied
