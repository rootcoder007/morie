"""Tests for rgzcr.rangayyan_zero_crossing.

Spec: Rangayyan & Krishnan (2024) Sec 5.6.2 "Zero-crossing rate", p.285 --
"the number of times the signal crosses the reference within a specified
interval". Expected values below are counted by hand from that definition,
not read back from the implementation.
"""

import numpy as np
import pytest

from morie.fn.rgzcr import rangayyan_zero_crossing


def test_rgzcr_counts_every_sign_change():
    # [+,-,+,-,+] changes sign between every adjacent pair: 4 crossings over
    # n-1 = 4 intervals, so ZCR is exactly 1.0 -- the maximum attainable.
    x = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    r = rangayyan_zero_crossing(x)
    assert r["crossings"] == 4
    assert r["n"] == 5
    assert r["zcr"] == pytest.approx(1.0)


def test_rgzcr_constant_signal_never_crosses():
    r = rangayyan_zero_crossing(np.ones(50))
    assert r["crossings"] == 0
    assert r["zcr"] == pytest.approx(0.0)


def test_rgzcr_single_crossing():
    # one sign change, 5 samples -> 1/(5-1) = 0.25
    x = np.array([2.0, 1.0, -1.0, -2.0, -3.0])
    r = rangayyan_zero_crossing(x)
    assert r["crossings"] == 1
    assert r["zcr"] == pytest.approx(0.25)


def test_rgzcr_per_second_scales_with_fs():
    x = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    r = rangayyan_zero_crossing(x, fs=200.0)
    assert r["zcr_per_second"] == pytest.approx(r["zcr"] * 200.0)


def test_rgzcr_rises_with_frequency():
    # Sec 5.6.2: "ZCR increases as the high-frequency content increases."
    t = np.arange(1000) / 1000.0
    slow = rangayyan_zero_crossing(np.sin(2 * np.pi * 5 * t))["zcr"]
    fast = rangayyan_zero_crossing(np.sin(2 * np.pi * 50 * t))["zcr"]
    assert fast > slow
