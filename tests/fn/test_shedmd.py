"""Tests for shedmd.viral_shedding_model (piecewise log10-linear curve)."""

import pytest

from morie.fn.shedmd import viral_shedding_model


DAYS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
LOG = [2.0, 3.0, 4.0, 5.0, 5.1, 4.9, 4.2, 3.4, 2.6, 1.8]


def test_shedmd_basic():
    """Each segment is an OLS fit on log10 load: an exact 1 log/day rise
    before the peak, the plateau mean, and an exact -0.8 log/day decay."""
    r = viral_shedding_model(DAYS, [10 ** v for v in LOG], 3.5, 5.5)
    assert r["rise_slope"] == pytest.approx(1.0, rel=1e-12)
    assert r["rise_intercept"] == pytest.approx(2.0, rel=1e-12)
    assert r["plateau_level"] == pytest.approx((5.1 + 4.9) / 2, rel=1e-12)
    assert r["decay_slope"] == pytest.approx(-0.8, rel=1e-12)
    assert r["decay_intercept"] == pytest.approx(4.2 + 0.8 * 6, rel=1e-12)
    assert (r["peak_day"], r["peak_load"]) == (4, pytest.approx(5.1, rel=1e-12))
    assert (r["n_rise"], r["n_plateau"], r["n_decay"]) == (4, 2, 4)


def test_shedmd_edge():
    """Non-positive loads and t_peak >= t_plateau raise."""
    with pytest.raises(ValueError):
        viral_shedding_model(DAYS, [0.0] + [10.0] * 9, 3.5, 5.5)
    with pytest.raises(ValueError):
        viral_shedding_model(DAYS, [10 ** v for v in LOG], 5.5, 3.5)
