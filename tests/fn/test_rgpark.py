"""Tests for bsaphys.rangayyan_parkinson_multimodal (sec. 10.14)."""

import math

import pytest

from morie.fn.bsaphys import rangayyan_freq_domain_feat, rangayyan_parkinson_multimodal


FS = 200.0
TREMOR = [math.sin(2 * math.pi * 5 * t / FS) + 0.1 * math.sin(2 * math.pi * 30 * t / FS) for t in range(800)]
STEADY = [math.sin(2 * math.pi * 20 * t / FS) + 0.1 * math.sin(2 * math.pi * 1 * t / FS) for t in range(800)]


def test_rgpark_basic():
    """Each modality's tremor index is the fraction of its PSD in the
    3-7 Hz band, the same fraction the PSD-feature routine reports for
    that band; a 5 Hz dominated EMG and gait are flagged."""
    r = rangayyan_parkinson_multimodal(STEADY, TREMOR, TREMOR, FS)
    ref = rangayyan_freq_domain_feat(TREMOR, FS, bands=[(3.0, 7.0)])["band_power_fraction"][0][2]
    assert r["emg_tremor_fraction"] == pytest.approx(ref, rel=1e-9)
    assert r["gait_tremor_fraction"] == pytest.approx(ref, rel=1e-9)
    assert r["emg_tremor_freq_hz"] == pytest.approx(5.0, abs=FS / 1024)
    assert r["tremor_present"] in (True, 1, 1.0)


def test_rgpark_edge():
    """No 3-7 Hz power: no tremor flagged."""
    r = rangayyan_parkinson_multimodal(STEADY, STEADY, STEADY, FS)
    assert r["emg_tremor_fraction"] < 0.05
    assert r["tremor_present"] in (False, 0, 0.0)
