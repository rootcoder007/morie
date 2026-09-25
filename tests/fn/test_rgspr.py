"""Tests for bsaqrs.rangayyan_spectral_power_ratio (sec. 8.12)."""

import math
import statistics

import pytest

from morie.fn.bsaqrs import rangayyan_spectral_power_ratio


def _rr(f_mod, n=300):
    return [0.8 + 0.05 * math.sin(2 * math.pi * f_mod * k * 0.8) for k in range(n)]


def test_rgspr_basic():
    """LF/HF is the ratio of the band powers (Task Force bands 0.04-0.15
    and 0.15-0.4 Hz); a 0.1 Hz modulation lands in LF, a 0.3 Hz one in
    HF; rrvar is the sample variance of the RR series."""
    lo = rangayyan_spectral_power_ratio(_rr(0.1))
    hi = rangayyan_spectral_power_ratio(_rr(0.3))
    assert lo["lfhf"] == pytest.approx(lo["lf"] / lo["hf"], rel=1e-12)
    assert lo["lfhf"] > 100 and hi["lfhf"] < 0.01
    assert lo["lfpct"] > 99 and hi["hfpct"] > 99
    assert lo["rrvar"] == pytest.approx(statistics.variance(_rr(0.1)), rel=1e-9)
    assert lo["n"] == 300


def test_rgspr_edge():
    """Unknown band conventions and non-positive intervals raise."""
    with pytest.raises(ValueError):
        rangayyan_spectral_power_ratio(_rr(0.1), bands="custom")
    with pytest.raises(ValueError):
        rangayyan_spectral_power_ratio([0.8, -0.1, 0.8] * 20)
