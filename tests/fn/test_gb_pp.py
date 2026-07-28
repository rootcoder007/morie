"""Tests for gb_pp (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb_pp import gibbons_pp_plot


def test_gb_pp_basic():
    from scipy import stats
    rng = np.random.default_rng(4)
    x = rng.standard_normal(150)
    assert gibbons_pp_plot(x)["max_departure"] == pytest.approx(
        stats.kstest(x, "norm").statistic, abs=1e-12)


def test_gb_pp_edge():
    with pytest.raises(ValueError):
        gibbons_pp_plot([1.0])
