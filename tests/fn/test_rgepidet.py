"""Tests for bsaphys.rangayyan_epilepsy_detect (sec. 8.17)."""

import math

import pytest

from morie.fn.bsaphys import rangayyan_epilepsy_detect


FS = 100.0


def _eeg(seizure=(6, 8)):
    """Ten 1 s epochs of alpha; the seizure epochs carry a large 3 Hz
    rhythm, the slow-wave signature the detector looks for."""
    out = []
    for t in range(1000):
        e = t // 100
        v = math.sin(2 * math.pi * 10 * t / FS) + 0.1 * math.sin(1.3 * t)
        if seizure[0] <= e < seizure[1]:
            v += 5.0 * math.sin(2 * math.pi * 3 * t / FS)
        out.append(v)
    return out


def test_rgepidet_basic():
    """The seizure epochs, and only they, are flagged, and the reported
    interval covers them."""
    r = rangayyan_epilepsy_detect(_eeg(), FS, baseline_epochs=4)
    assert r["seizure_detected"] in (True, 1, 1.0)
    assert r["n_epochs"] == 10
    assert r["n_flagged"] == 2
    lo, hi = r["seizure_intervals_s"][0]
    assert lo == pytest.approx(6.0, abs=1e-9) and hi == pytest.approx(8.0, abs=1e-9)


def test_rgepidet_edge():
    """Pure alpha: nothing flagged; fewer than two epochs raises."""
    r = rangayyan_epilepsy_detect(_eeg(seizure=(0, 0)), FS)
    assert r["n_flagged"] == 0
    with pytest.raises(ValueError):
        rangayyan_epilepsy_detect(_eeg()[:150], FS)
