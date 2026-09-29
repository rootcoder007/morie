"""Tests for morie.fn.spcgm: Parseval and the peak frequency."""

import math

from morie.fn.spcgm import spcgm

X = [math.sin(2 * math.pi * 0.125 * n) + 0.5 * math.cos(2 * math.pi * 0.3125 * n) for n in range(96)]


def test_density_integrates_to_segment_power():
    r = spcgm(X, fs=2.0, nperseg=32, noverlap=16, window="boxcar")
    # boxcar, one segment: sum_k S_k * fs / nperseg equals the mean power of the detrended segment
    seg = X[:32]
    m = sum(seg) / 32
    power = sum((v - m) ** 2 for v in seg) / 32
    col = [row[0] for row in r.value]
    assert abs(sum(col) * 2.0 / 32 - power) < 1e-12
    assert r.extra["times"][:2] == [8.0, 16.0]
    assert r.extra["frequencies"][4] == 0.25


def test_peak_and_zero_padding():
    r = spcgm(X, nperseg=32, noverlap=0, nfft=64)
    col = [row[1] for row in r.value]
    assert max(range(len(col)), key=lambda k: col[k]) == 8
    assert len(col) == 33
