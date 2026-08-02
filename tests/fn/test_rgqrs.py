"""Tests for rgqrs.rangayyan_qrs_detect.

Spec: Rangayyan & Krishnan, Biomedical Signal Analysis, 3rd ed. (IEEE Press /
Wiley, 2024), Sec. 4.3.2 "The Pan-Tompkins algorithm for QRS detection",
p.220, eqs (4.7), (4.8), (4.14), (4.15); after Pan & Tompkins (1985), IEEE
Trans. Biomed. Eng. BME-32(3):230-236.

The docstring had cited Ch. 6. The algorithm is Sec. 4.3.2.

The book prints no numeric worked example for the detector, so the anchor here
is a synthetic ECG whose R-peak sample indices are known BY CONSTRUCTION: the
test builds the beats, so the ground truth needs no transcription and no
appeal to the implementation.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.rgqrs import rangayyan_qrs_detect


def _synthetic_ecg(fs=360.0, bpm=60.0, n_beats=12, noise=0.0, seed=0):
    """Build an ECG-like trace and return (signal, true R-peak indices).

    Each beat is a sharp biphasic QRS plus a broad low-frequency T wave, so a
    detector that merely finds maxima would latch onto the T wave; the
    bandpass and squaring stages are what separate them.
    """
    rng = np.random.default_rng(seed)
    period = int(round(60.0 / bpm * fs))
    n = period * n_beats + period // 2
    x = np.zeros(n)
    peaks = []
    for b in range(n_beats):
        r = period // 2 + b * period
        peaks.append(r)
        w = int(round(0.040 * fs))                      # ~40 ms QRS
        t = np.arange(-w, w + 1)
        x[r - w : r + w + 1] += 1.6 * np.exp(-((t / (w / 2.2)) ** 2)) \
                                - 0.4 * np.exp(-(((t - w) / (w / 1.5)) ** 2))
        tw = int(round(0.16 * fs))                      # broad T wave, later
        ts = r + int(round(0.22 * fs))
        if ts + tw < n:
            u = np.arange(-tw, tw + 1)
            seg = 0.35 * np.exp(-((u / (tw / 1.8)) ** 2))
            lo, hi = ts - tw, ts + tw + 1
            x[lo:hi] += seg[: hi - lo]
    if noise:
        x += noise * rng.standard_normal(n)
    return x, np.asarray(peaks)


def test_detects_every_beat_in_a_clean_synthetic_ecg():
    """Ground truth is known by construction, not transcribed or inferred."""
    fs = 360.0
    x, true_peaks = _synthetic_ecg(fs=fs, bpm=60.0, n_beats=12)
    got = rangayyan_qrs_detect(x, fs=fs)["r_peaks"]
    assert got.size == true_peaks.size, f"expected {true_peaks.size} beats, got {got.size}"
    # Within 50 ms of truth -- the refinement window the detector itself uses.
    tol = int(round(0.05 * fs))
    assert np.all(np.abs(got - true_peaks) <= tol), \
        f"max offset {np.max(np.abs(got - true_peaks))} samples > {tol}"


def test_ignores_t_waves():
    """The T wave is larger in duration and would fool a naive peak finder.

    Detecting exactly n_beats peaks -- not 2*n_beats -- is the evidence that
    the bandpass and squaring stages are doing their job.
    """
    fs = 250.0
    x, true_peaks = _synthetic_ecg(fs=fs, bpm=72.0, n_beats=10)
    assert rangayyan_qrs_detect(x, fs=fs)["r_peaks"].size == true_peaks.size


def test_survives_additive_noise():
    fs = 360.0
    x, true_peaks = _synthetic_ecg(fs=fs, bpm=60.0, n_beats=12, noise=0.05, seed=3)
    got = rangayyan_qrs_detect(x, fs=fs)["r_peaks"]
    assert got.size == true_peaks.size


def test_identity_heart_rate_matches_the_construction():
    """60 bpm in, 60 bpm out. HR is derived from the RR intervals, so this
    closes the loop from detection through to the reported rate."""
    fs = 360.0
    for bpm in (50.0, 60.0, 80.0):
        x, _ = _synthetic_ecg(fs=fs, bpm=bpm, n_beats=14)
        got = rangayyan_qrs_detect(x, fs=fs)["heart_rate_bpm"]
        assert abs(got - bpm) < 1.0, f"expected ~{bpm} bpm, got {got}"


def test_identity_rr_intervals_are_consistent_with_the_peaks():
    """rr_intervals_ms must be the successive differences of r_peaks in ms."""
    fs = 360.0
    x, _ = _synthetic_ecg(fs=fs, bpm=75.0, n_beats=10)
    res = rangayyan_qrs_detect(x, fs=fs)
    expect = np.diff(res["r_peaks"]) * (1000.0 / fs)
    assert np.allclose(res["rr_intervals_ms"], expect)


def test_identity_amplitude_scale_invariance():
    """Detection is threshold-relative (0.3 x max of the integrated signal),
    so scaling the ECG must not change which samples are detected."""
    fs = 360.0
    x, _ = _synthetic_ecg(fs=fs, bpm=60.0, n_beats=12)
    base = rangayyan_qrs_detect(x, fs=fs)["r_peaks"]
    for a in (1000.0, 0.001, -1.0):
        assert np.array_equal(rangayyan_qrs_detect(a * x, fs=fs)["r_peaks"], base)


def test_rejects_a_series_too_short_to_hold_a_beat():
    """The generated tests passed 5 samples and 1 sample, and the failure came
    out of scipy as "The length of the input vector x must be greater than
    padlen" -- a message that says nothing about ECG. The function was right
    to refuse; it now refuses in its own words."""
    with pytest.raises(ValueError, match="at least .* samples"):
        rangayyan_qrs_detect(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    with pytest.raises(ValueError, match="at least .* samples"):
        rangayyan_qrs_detect(np.array([42.0]))


def test_rejects_non_positive_sampling_rate():
    x, _ = _synthetic_ecg()
    with pytest.raises(ValueError, match="`fs` must be positive"):
        rangayyan_qrs_detect(x, fs=0.0)


def test_returns_documented_keys():
    x, _ = _synthetic_ecg()
    res = rangayyan_qrs_detect(x)
    for key in ("r_peaks", "rr_intervals_ms", "heart_rate_bpm", "integrated", "fs"):
        assert key in res
