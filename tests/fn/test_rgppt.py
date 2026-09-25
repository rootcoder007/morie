"""Tests for bsaphys.rangayyan_point_process (sec. 7.3)."""

import statistics

import pytest

from morie.fn.bsaphys import rangayyan_point_process


EV = [0.0, 0.05, 0.09, 0.15, 0.2, 0.24, 0.31]


def test_rgppt_basic():
    """Inter-event intervals: mean, sample SD, CV; the repetition rates
    1/IPI give mu_r, sigma_r and CV_r = sigma_r / mu_r."""
    ipi = [b - a for a, b in zip(EV, EV[1:])]
    rate = [1 / v for v in ipi]
    r = rangayyan_point_process(EV)
    assert r["mean_ipi_s"] == pytest.approx(statistics.fmean(ipi), rel=1e-12)
    assert r["sd_ipi_s"] == pytest.approx(statistics.stdev(ipi), rel=1e-12)
    assert r["cv_ipi"] == pytest.approx(statistics.stdev(ipi) / statistics.fmean(ipi), rel=1e-12)
    assert r["mean_rate_pps"] == pytest.approx(statistics.fmean(rate), rel=1e-12)
    assert r["sd_rate_pps"] == pytest.approx(statistics.stdev(rate), rel=1e-12)
    assert r["cv_rate"] == pytest.approx(statistics.stdev(rate) / statistics.fmean(rate), rel=1e-12)
    assert (r["n_events"], r["n_intervals"]) == (7, 6)
    assert r["median_ipi_s"] == pytest.approx(statistics.median(ipi), rel=1e-12)


def test_rgppt_edge():
    """A perfectly periodic train has zero CV; unsorted times raise."""
    r = rangayyan_point_process([0.0, 0.1, 0.2, 0.3, 0.4])
    assert r["cv_ipi"] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        rangayyan_point_process([0.0, 0.2, 0.1])
