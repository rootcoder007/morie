"""Tests for ksr044 (Kosorok shelf)."""

import numpy as np
import pytest

from morie.fn.ksr044 import kosorok_ch2_quantile_taylor_bounds


def test_ksr044_basic():
    from scipy import stats
    h = lambda z: 0.1 * stats.norm.pdf(z)
    out = kosorok_ch2_quantile_taylor_bounds(stats.norm.cdf, h, t_n=0.01, p=0.6)
    xi = stats.norm.ppf(0.6)
    assert out["implied_derivative"] == pytest.approx(-h(xi) / stats.norm.pdf(xi),
                                                      rel=1e-3)


def test_ksr044_edge():
    from scipy import stats
    with pytest.raises(ValueError):
        kosorok_ch2_quantile_taylor_bounds(stats.norm.cdf, lambda z: 0.0,
                                           t_n=-1.0, p=0.6)
