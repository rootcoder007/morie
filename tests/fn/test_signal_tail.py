"""Equivalence tests: _signal_core spectral tail vs frozen scipy
anchors (versions recorded in oracle_anchors.json; scipy itself is not
imported)."""
import json
import pathlib

import pytest

from morie.fn import _array_core as np
from morie.fn import _signal_core as mg

A = json.loads(pathlib.Path(__file__).with_name(
    "oracle_anchors.json").read_text())


def _sig(n=600):
    rng = np.random.default_rng(5)
    return np.sin(2 * np.pi * 0.1 * np.arange(n)) \
        + 0.4 * rng.normal(0, 1, n)


def test_welch_matches():
    xa = _sig()
    fg, pg = mg.welch(xa.tolist(), fs=2.0, nperseg=128)
    assert fg.tolist() == pytest.approx(A["welch"]["f"])
    assert pg.tolist() == pytest.approx(A["welch"]["p"], rel=1e-8)


def test_coherence_matches():
    xa = _sig()
    rng = np.random.default_rng(6)
    ya = np.cos(2 * np.pi * 0.1 * np.arange(600)) \
        + 0.4 * rng.normal(0, 1, 600)
    fg, cg = mg.coherence(xa.tolist(), ya.tolist(), fs=2.0, nperseg=128)
    assert cg.tolist() == pytest.approx(A["coherence"], rel=1e-6)


def test_hilbert_matches():
    xa = _sig(256)
    hg = mg.hilbert(xa.tolist()).tolist()
    for i in (0, 50, 100, 255):
        re, im = A["hilbert"][str(i)]
        assert hg[i].real == pytest.approx(re, rel=1e-8, abs=1e-10)
        assert hg[i].imag == pytest.approx(im, rel=1e-8, abs=1e-10)


def test_fftconvolve_all_modes():
    a, b = [1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 0.25]
    for mode in ("full", "same", "valid"):
        g = mg.fftconvolve(a, b, mode=mode).tolist()
        assert g == pytest.approx(A["fftconvolve"][mode], abs=1e-10)


def test_find_peaks_matches():
    x = [0, 2, 0, 3, 0, 1, 0]
    pg, _ = mg.find_peaks(x, height=1.5)
    assert [int(v) for v in pg.tolist()] == A["find_peaks"]


def test_savgol_interior_matches():
    xa = _sig(99)
    got = mg.savgol_filter(xa.tolist(), 11, 3).tolist()
    # interior points exact; edges use mirror padding vs scipy 'interp'
    assert got[20:80] == pytest.approx(A["savgol_20_80"], rel=1e-9)


def test_medfilt_and_detrend():
    x = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    assert mg.medfilt(x, 3).tolist() == A["medfilt"]
    xa = _sig(50)
    assert mg.detrend(xa.tolist()).tolist() == pytest.approx(
        A["detrend50"], rel=1e-9, abs=1e-12)


def test_stft_matches():
    xa = _sig()
    fg, tg, zg = mg.stft(xa.tolist(), fs=2.0, nperseg=64)
    for k in (2, 5, 20):
        for t in (0, 2):
            assert abs(zg[k][t]) == pytest.approx(
                A["stft_absz"]["%d_%d" % (k, t)], rel=1e-7, abs=1e-12)


def test_spectrogram_matches():
    xa = _sig()
    fg, tg, G = mg.spectrogram(xa.tolist(), fs=2.0, nperseg=128)
    W = A["spectrogram"]["W"]
    got = G.tolist()
    assert len(got) == len(W)
    for r1, r2 in zip(got, W):
        assert r1 == pytest.approx(r2, rel=1e-8, abs=1e-12)
    assert tg.tolist() == pytest.approx(A["spectrogram"]["t"])
