"""Equivalence tests: morie.fn._signal_core vs scipy.signal."""

import math

import pytest

import pytest

sp_sig = pytest.importorskip(
    "scipy.signal",
    reason="oracle test: compares the native cores against real scipy where it exists")

from morie.fn import _signal_core as ms


def _sig(n=400):
    return [math.sin(2 * math.pi * 0.05 * i)
            + 0.5 * math.sin(2 * math.pi * 0.2 * i + 0.3)
            + 0.1 * math.cos(2 * math.pi * 0.37 * i) for i in range(n)]


def test_butter_ba_lowpass_matches():
    for n in (2, 3, 4, 5):
        b, a = ms.butter(n, 0.2, btype="low", output="ba")
        bw, aw = sp_sig.butter(n, 0.2, btype="low", output="ba")
        assert b == pytest.approx(list(bw), rel=1e-9, abs=1e-12)
        assert a == pytest.approx(list(aw), rel=1e-9, abs=1e-12)


def test_butter_ba_highpass_matches():
    for n in (2, 4):
        b, a = ms.butter(n, 0.3, btype="high", output="ba")
        bw, aw = sp_sig.butter(n, 0.3, btype="high", output="ba")
        assert b == pytest.approx(list(bw), rel=1e-9, abs=1e-12)
        assert a == pytest.approx(list(aw), rel=1e-9, abs=1e-12)


def test_butter_ba_bandpass_matches():
    b, a = ms.butter(3, [0.1, 0.4], btype="band", output="ba")
    bw, aw = sp_sig.butter(3, [0.1, 0.4], btype="band", output="ba")
    assert b == pytest.approx(list(bw), rel=1e-8, abs=1e-11)
    assert a == pytest.approx(list(aw), rel=1e-8, abs=1e-11)


def test_lfilter_matches():
    x = _sig()
    b, a = sp_sig.butter(4, 0.2)
    got = ms.lfilter(list(b), list(a), x).tolist()
    want = sp_sig.lfilter(b, a, x).tolist()
    assert got == pytest.approx(want, rel=1e-9, abs=1e-12)


def test_lfilter_zi_matches():
    b, a = sp_sig.butter(4, 0.2)
    got = ms.lfilter_zi(list(b), list(a))
    want = sp_sig.lfilter_zi(b, a).tolist()
    assert got == pytest.approx(want, rel=1e-8, abs=1e-11)


def test_filtfilt_matches():
    x = _sig()
    for n, wn in ((2, 0.15), (4, 0.2), (5, 0.35)):
        b, a = sp_sig.butter(n, wn)
        got = ms.filtfilt(list(b), list(a), x).tolist()
        want = sp_sig.filtfilt(b, a, x).tolist()
        assert got == pytest.approx(want, rel=1e-7, abs=1e-9)


def test_sosfiltfilt_matches_low_high_band():
    x = _sig()
    cases = [(4, 0.2, "low"), (4, 0.3, "high"), (3, [0.1, 0.4], "band")]
    for n, wn, bt in cases:
        sos_w = sp_sig.butter(n, wn, btype=bt, output="sos")
        sos_g = ms.butter(n, wn, btype=bt, output="sos")
        want = sp_sig.sosfiltfilt(sos_w, x).tolist()
        got = ms.sosfiltfilt(sos_g, x).tolist()
        assert got == pytest.approx(want, rel=1e-6, abs=1e-8)
        # our sos through scipy's sosfiltfilt must also agree ->
        # proves the sections themselves are a valid factorization
        cross = sp_sig.sosfiltfilt(sos_g.tolist(), x).tolist()
        assert cross == pytest.approx(want, rel=1e-6, abs=1e-8)


def test_sosfilt_zi_steady_state():
    sos = ms.butter(4, 0.2, output="sos")
    zi = ms.sosfilt_zi(sos)
    y, _ = ms.sosfilt(sos, [1.0] * 50,
                      zi=[[v for v in z] for z in zi])
    # steady-state ic -> unit step passes through at DC gain immediately
    assert y.tolist()[0] == pytest.approx(y.tolist()[-1], rel=1e-9)
