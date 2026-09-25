"""Tests for stratm.stratmean (Cochran 1977, Theorem 5.3)."""

import math
import statistics

import pytest

from morie.fn.stratm import stratified_mean, stratmean


def _data():
    y = [3.1, 2.7, 3.9, 3.3, 5.2, 4.8, 6.1, 5.5, 4.9, 1.2, 0.8, 1.9]
    h = [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3]
    return y, h, [40, 100, 20]


def test_stratm_basic():
    """ybar_st = sum W_h ybar_h and V = sum W_h^2 (1 - f_h) s_h^2/n_h,
    s_h^2 the stratum sample variance and f_h = n_h/N_h; the interval
    is ybar_st -+ z se."""
    y, h, Nh = _data()
    N = sum(Nh)
    groups = [[v for v, g in zip(y, h) if g == s] for s in (1, 2, 3)]
    W = [v / N for v in Nh]
    est = sum(w * statistics.fmean(g) for w, g in zip(W, groups))
    var = sum(w * w * (1 - len(g) / Nk) * statistics.variance(g) / len(g) for w, g, Nk in zip(W, groups, Nh))
    r = stratified_mean(y, h, Nh, level=0.9)
    assert r["estimate"] == pytest.approx(est, abs=1e-14)
    assert r["se"] == pytest.approx(math.sqrt(var), rel=1e-13)
    z = statistics.NormalDist().inv_cdf(0.95)
    assert r["ci_lower"] == pytest.approx(est - z * math.sqrt(var), rel=1e-9)
    assert r["unweighted_mean"] == pytest.approx(statistics.fmean(y), abs=1e-15)
    assert r["Wh"] == pytest.approx(W, abs=1e-15)


def test_stratm_edge():
    """Proportional allocation makes ybar_st the sample mean; zero-based
    labels, a stratum of one and an oversize sample raise."""
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0]
    r = stratmean(y, [1, 1, 1, 2, 2, 2], [30, 30])
    assert r["estimate"] == pytest.approx(sum(y) / 6, abs=1e-15)
    with pytest.raises(ValueError):
        stratmean(y, [0, 0, 0, 1, 1, 1], [30, 30])
    with pytest.raises(ValueError):
        stratmean(y, [1, 1, 1, 1, 1, 2], [30, 30])
    with pytest.raises(ValueError):
        stratmean(y, [1, 1, 1, 2, 2, 2], [30, 2])
