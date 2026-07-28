"""Tests for gb1131t (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb1131t import gibbons_spearman_ties


def test_gb1131t_basic():
    from scipy import stats
    rng = np.random.default_rng(1)
    x = np.round(rng.standard_normal(30), 1); y = np.round(x + rng.standard_normal(30), 1)
    assert gibbons_spearman_ties(x, y)["r_s"] == pytest.approx(
        stats.spearmanr(x, y).statistic, abs=1e-10)


def test_gb1131t_edge():
    with pytest.raises(ValueError):
        gibbons_spearman_ties([1, 1, 1], [1, 2, 3])
