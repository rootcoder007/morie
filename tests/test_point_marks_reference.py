"""Berman-Diggle bandwidth, mark correlation, mark variogram, mark dependence test.

Checked against spatstat.explore 3.8 (bw.diggle exactly; markcorr and
markvario with method "density" to 4e-15, including stats::density's
binning); tests/cross/test-morie_vs_spatstat.R repeats that in R.
"""

import math

import pytest

from morie.fn.bwdigg import bandwidth_diggle
from morie.fn.mkcorr import mark_correlation, mark_variogram
from morie.fn.mkdep import mark_dependence_test

PTS = [
    (0.1, 0.2),
    (0.4, 0.8),
    (0.35, 0.3),
    (0.8, 0.6),
    (0.7, 0.15),
    (0.55, 0.5),
    (0.2, 0.65),
    (0.9, 0.9),
    (0.15, 0.95),
    (0.6, 0.85),
]
MK = [1.0, 2.0, 1.5, 3.0, 0.5, 2.5, 1.0, 2.0, 0.8, 1.2]
W = (0, 1, 0, 1)


def test_bandwidth_diggle_documented_and_grid():
    r = bandwidth_diggle(PTS, W)
    assert (round(r.sigma, 6), r.at_boundary) == (0.062378, True)
    # rmax = min(1/4, sqrt(1000/(10 pi))) = 0.25, r_k = 0.25 k / 511, kept while r_k <= 0.125
    assert len(r.h) == 256
    assert r.h[-1] == pytest.approx(0.25 * 255 / 511 / 2, abs=1e-15)
    assert r.sigma == r.h[-1]
    finite = [c for c in r.criterion if c == c]
    assert min(finite) == r.criterion[r.h.index(r.sigma)]
    assert r.lambda_hat == 10.0


def test_mark_correlation_constant_marks_is_one():
    k = mark_correlation(PTS, [2.0] * 10, W).k
    assert all(abs(v - 1.0) < 1e-12 for v in k if v == v)


def test_mark_correlation_documented_value_and_variogram_constant():
    assert round(mark_correlation(PTS, MK, W).k[256], 6) == 0.998959
    assert round(mark_variogram(PTS, MK, W).gamma[256], 6) == 0.32
    g = mark_variogram(PTS, [2.0] * 10, W).gamma
    assert all(abs(v) < 1e-12 for v in g if v == v)
    assert mark_variogram(PTS, MK, W).theo == pytest.approx(sum((v - sum(MK) / 10) ** 2 for v in MK) / 9, abs=1e-12)


def test_mark_correlation_rejects_negative_marks():
    with pytest.raises(ValueError):
        mark_correlation(PTS, [-1.0] + MK[1:], W)


def test_mark_dependence_test_documented_and_deterministic():
    t = mark_dependence_test(PTS, MK, W, nsim=19)
    assert (round(t.statistic, 6), t.p_value) == (0.161439, 1.0)
    k = [v for v in t.extra["k"] if v == v]
    assert t.statistic == pytest.approx(max(abs(v - 1.0) for v in k), abs=1e-12)
    again = mark_dependence_test(PTS, MK, W, nsim=19)
    assert again.extra["simulated"] == t.extra["simulated"]
    d = mark_dependence_test(PTS, MK, W, nsim=9, statistic="dclf")
    assert 0.0 < d.p_value <= 1.0 and math.isfinite(d.statistic)
