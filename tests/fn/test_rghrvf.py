"""Tests for bsaqrs.rangayyan_hrv_freq_domain (Rangayyan 2024 sec. 8.12)."""

import math

import pytest

from morie.fn.bsaqrs import rangayyan_hrv_freq_domain


RR = [0.8 + 0.05 * math.sin(2 * math.pi * 0.1 * 0.8 * k) + 0.02 * math.sin(2 * math.pi * 0.3 * 0.8 * k)
      for k in range(400)]


def test_rghrvf_basic():
    """Task Force bands VLF <= 0.04, LF 0.04-0.15, HF 0.15-0.4 Hz; the
    band powers partition the total, the percentages sum to 100 and
    LF/HF is their ratio.  A 0.1 Hz RR modulation of amplitude 0.05 s
    against 0.02 s at 0.3 Hz makes LF dominate."""
    r = rangayyan_hrv_freq_domain(RR)
    assert r["limits"] == {"vlf": (0.0, 0.04), "lf": (0.04, 0.15), "hf": (0.15, 0.4)}
    assert r["vlf"] + r["lf"] + r["hf"] == pytest.approx(r["total"], rel=1e-12)
    assert r["vlfpct"] + r["lfpct"] + r["hfpct"] == pytest.approx(100.0, rel=1e-12)
    assert r["lfhf"] == pytest.approx(r["lf"] / r["hf"], rel=1e-12)
    assert r["lf"] > 5 * r["hf"] > 0
    assert rangayyan_hrv_freq_domain(RR, bands="bianchi")["limits"]["hf"] == (0.18, 0.4)


def test_rghrvf_edge():
    """Non-positive RR intervals are rejected."""
    with pytest.raises(ValueError):
        rangayyan_hrv_freq_domain([0.8, -0.1, 0.9] * 20)
