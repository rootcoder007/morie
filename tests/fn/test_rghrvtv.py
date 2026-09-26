"""Tests for bsatf.rangayyan_hrv_time_varying (Rangayyan and Krishnan 2024 sec. 8.12)."""

import math

import pytest

from morie.fn.bsatf import rangayyan_hrv_time_varying


RR = [0.8 + 0.05 * math.sin(2 * math.pi * 0.1 * 0.8 * k) + 0.02 * math.sin(2 * math.pi * 0.3 * 0.8 * k)
      for k in range(400)]


def test_rghrvtv_basic():
    """Per window: LF/HF is the ratio of the band powers and the
    percentages are shares of the window total; mean RR 0.8 s is 75 bpm;
    the tachogram is resampled at 4 Hz over the 320 s record."""
    r = rangayyan_hrv_time_varying(RR)
    assert r["mean_rr"] == pytest.approx(sum(RR) / len(RR), rel=1e-12)
    assert r["mean_hr"] == pytest.approx(60.0 / r["mean_rr"], rel=1e-12)
    assert len(r["resampled"]) == 1280
    for lf, hf, ratio, lp, tot in zip(r["lf"], r["hf"], r["lf_hf_ratio"], r["lf_percent"], r["total_power"]):
        assert ratio == pytest.approx(lf / hf, rel=1e-12)
        assert lp == pytest.approx(100 * lf / tot, rel=1e-12)
    assert sum(r["lf"]) > sum(r["hf"])


def test_rghrvtv_edge():
    """Non-positive RR intervals are rejected."""
    with pytest.raises(ValueError):
        rangayyan_hrv_time_varying([0.8, 0.0] * 100)
