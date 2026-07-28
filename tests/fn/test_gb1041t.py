"""Tests for gb1041t (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb1041t import gibbons_kw_ties


def test_gb1041t_basic():
    from scipy import stats
    rng = np.random.default_rng(8)
    g = [np.round(rng.standard_normal(15) + d, 0) for d in (0, 0.5, 1.0)]
    assert gibbons_kw_ties(g)["H"] == pytest.approx(stats.kruskal(*g).statistic, abs=1e-10)


def test_gb1041t_edge():
    with pytest.raises(ValueError):
        gibbons_kw_ties([[1.0, 2.0]])  # one group
