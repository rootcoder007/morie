"""Equivalence tests: _signal_core spectral tail vs scipy.signal."""

import numpy as np
import pytest

import scipy.signal as sg

from morie.fn import _signal_core as mg


def _sig(n=600):
    rng = np.random.default_rng(5)
    return np.sin(2 * np.pi * 0.1 * np.arange(n)) \
        + 0.4 * rng.normal(0, 1, n)


def test_welch_matches():
    xa = _sig()
    fg, pg = mg.welch(xa.tolist(), fs=2.0, nperseg=128)
    fw, pw = sg.welch(xa, fs=2.0, nperseg=128)
    assert fg.tolist() == pytest.approx(fw.tolist())
    assert pg.tolist() == pytest.approx(pw.tolist(), rel=1e-8)


def test_coherence_matches():
    xa = _sig()
    rng = np.random.default_rng(6)
    ya = np.cos(2 * np.pi * 0.1 * np.arange(600)) \
        + 0.4 * rng.normal(0, 1, 600)
    fg, cg = mg.coherence(xa.tolist(), ya.tolist(), fs=2.0, nperseg=128)
    fw, cw = sg.coherence(xa, ya, fs=2.0, nperseg=128)
    assert cg.tolist() == pytest.approx(cw.tolist(), rel=1e-6)


def test_hilbert_matches():
    xa = _sig(256)
    hg = mg.hilbert(xa.tolist()).tolist()
    hw = sg.hilbert(xa)
    for i in (0, 50, 100, 255):
        assert hg[i].real == pytest.approx(hw[i].real, rel=1e-8, abs=1e-10)
        assert hg[i].imag == pytest.approx(hw[i].imag, rel=1e-8, abs=1e-10)


def test_fftconvolve_all_modes():
    a, b = [1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 0.25]
    for mode in ("full", "same", "valid"):
        g = mg.fftconvolve(a, b, mode=mode).tolist()
        w = sg.fftconvolve(np.array(a), np.array(b), mode=mode).tolist()
        assert g == pytest.approx(w, abs=1e-10)


def test_find_peaks_matches():
    x = [0, 2, 0, 3, 0, 1, 0]
    pg, _ = mg.find_peaks(x, height=1.5)
    pw, _ = sg.find_peaks(np.array(x), height=1.5)
    assert [int(v) for v in pg.tolist()] == pw.tolist()


def test_savgol_interior_matches():
    xa = _sig(99)
    got = mg.savgol_filter(xa.tolist(), 11, 3).tolist()
    want = sg.savgol_filter(xa, 11, 3)
    # interior points exact; edges use mirror padding vs scipy 'interp'
    assert got[20:80] == pytest.approx(want[20:80].tolist(), rel=1e-9)


def test_medfilt_and_detrend():
    x = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    assert mg.medfilt(x, 3).tolist() == sg.medfilt(np.array(x), 3).tolist()
    xa = _sig(50)
    assert mg.detrend(xa.tolist()).tolist() == pytest.approx(
        sg.detrend(xa).tolist(), rel=1e-9, abs=1e-12)


def test_stft_matches():
    xa = _sig()
    fg, tg, zg = mg.stft(xa.tolist(), fs=2.0, nperseg=64)
    fw, tw, zw = sg.stft(xa, fs=2.0, nperseg=64, boundary=None,
                         padded=False)
    for k in (2, 5, 20):
        for t in (0, 2):
            assert abs(zg[k][t]) == pytest.approx(
                abs(zw[k][t]), rel=1e-7, abs=1e-12)


def test_spectrogram_matches():
    xa = _sig()
    fg, tg, G = mg.spectrogram(xa.tolist(), fs=2.0, nperseg=128)
    fw, tw, W = sg.spectrogram(xa, fs=2.0, nperseg=128)
    assert np.allclose(np.array(G.tolist()), np.array(W), rtol=1e-8)
    assert tg.tolist() == pytest.approx(tw.tolist())
