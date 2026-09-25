"""Tests for bsaphys.rangayyan_sleep_apnea_detect (sec. 10.13)."""

import math

import pytest

from morie.fn.bsaphys import rangayyan_sleep_apnea_detect


FS = 100.0
APNEA = (2, 4)


def _g(t, c, w):
    return math.exp(-((t - c) ** 2) / (2 * w * w))


ECG = [1.0 if i % 100 == 50 else 0.0 for i in range(6000)]
SPO2 = [97.0 - (6.0 * _g((i % 1000) / 1000.0, 0.5, 0.1) if i // 1000 in APNEA else 0.0) for i in range(6000)]
SNORE = [(3.0 if i // 1000 in APNEA else 1.0) * math.sin(0.9 * i) for i in range(6000)]


def test_rgsapdet_basic():
    """Per 10 s epoch: desaturation depth is max - min SpO2, snore RMS is
    the channel RMS, HR is 60 bpm from 1 s R-R; the score counts the
    criteria met (depth >= 4, snore >= 1.5 x median, LF >= 0.3) and an
    epoch is flagged at score >= 2.  Two flagged 10 s epochs in a
    60 s record are 120 events per hour, the "severe" AHI band."""
    r = rangayyan_sleep_apnea_detect(ECG, SPO2, SNORE, FS, epoch_s=10.0)
    rms = [math.sqrt(sum(v * v for v in SNORE[k * 1000:(k + 1) * 1000]) / 1000) for k in range(6)]
    med = sorted(rms)[3]
    for k, row in enumerate(r["epochs"]):
        seg = SPO2[k * 1000:(k + 1) * 1000]
        assert row["desat_depth_pct"] == pytest.approx(max(seg) - min(seg), abs=1e-12)
        assert row["snore_rms"] == pytest.approx(rms[k], rel=1e-12)
        assert row["mean_hr_bpm"] == pytest.approx(60.0, rel=1e-12)
        lf = row["hr_lf_fraction"]
        score = (row["desat_depth_pct"] >= 4.0) + (row["snore_rms"] >= 1.5 * med) + (lf is not None and lf >= 0.3)
        assert row["score"] == score
        assert row["epoch_flagged"] == (k in APNEA)
    assert r["n_flagged"] == 2
    assert r["events_per_hour"] == pytest.approx(2 / (60.0 / 3600.0), rel=1e-12)
    assert r["severity"] == "severe"


def test_rgsapdet_edge():
    """SpO2 outside 0-100 and fs below 100 Hz raise."""
    with pytest.raises(ValueError):
        rangayyan_sleep_apnea_detect(ECG, [101.0] * 6000, SNORE, FS, epoch_s=10.0)
    with pytest.raises(ValueError):
        rangayyan_sleep_apnea_detect(ECG, SPO2, SNORE, 50.0, epoch_s=10.0)
