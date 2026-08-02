# morie.fn -- function file (rootcoder007/morie)
"""Pan-Tompkins QRS detection -- Rangayyan & Krishnan Sec. 4.3.2, p.220."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_qrs_detect"]


def rangayyan_qrs_detect(x, fs=360.0):
    """Pan-Tompkins QRS detector.

    Pipeline (Pan & Tompkins 1985; Rangayyan & Krishnan Sec. 4.3.2, p.220):

    1.  Bandpass 5–15 Hz (Butterworth IIR, zero-phase).
    2.  Differentiate ``y[n] = (1/8)(2 x[n] + x[n-1] - x[n-3] - 2 x[n-4])``.
    3.  Square.
    4.  Moving-window integration (window ≈ 150 ms).
    5.  Adaptive threshold = 0.3 × max(integrated) with refractory
        ≈ 200 ms; peaks above threshold are R-peaks.

    Parameters
    ----------
    x : array-like
        Raw ECG.
    fs : float
        Sampling rate (Hz, default 360 -- MIT-BIH).

    Returns
    -------
    RichResult with keys ``r_peaks`` (sample indices), ``rr_intervals_ms``,
    ``heart_rate_bpm``, ``integrated``, ``fs``.

    References
    ----------
    Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm.
        *IEEE Transactions on Biomedical Engineering*, BME-32(3), 230-236.
    Rangayyan, R. M., & Krishnan, S. *Biomedical Signal Analysis*, 3rd ed.
        (IEEE Press / Wiley, 2024), Sec. 4.3.2 "The Pan-Tompkins algorithm
        for QRS detection", p.220, eqs (4.7), (4.8), (4.14), (4.15).

    Note: the docstring previously cited Ch. 6; the algorithm is Sec. 4.3.2 in
    the edition we hold.

    Verified against the book: the derivative here is eq (4.14) exactly,
    y(n) = (1/8)[2x(n) + x(n-1) - x(n-3) - 2x(n-4)], and the 150 ms
    integration window is eq (4.15)'s N = 30 samples at the paper's
    fs = 200 Hz.

    Note on two deliberate deviations from the original, both consequences of
    this being an OFFLINE detector rather than the real-time one Pan and
    Tompkins designed:

    * Bandpass. The original cascades the integer-coefficient recursive
      lowpass of eq (4.7)-(4.8) with a matching highpass, chosen so the
      filter can run in real time on 1985 hardware. This uses a zero-phase
      Butterworth bandpass (``sosfiltfilt``) over the same 5-15 Hz band.
      Zero-phase filtering is non-causal and cannot be done in real time, but
      it removes the group delay the original has to correct for.
    * Integration. Eq (4.15) is a causal moving average over the preceding N
      samples, which lags the QRS by about N/2. This convolves with
      ``mode="same"``, i.e. centred, so the integrated peak sits on the QRS
      rather than after it. Given the +/-50 ms refinement against the
      bandpassed signal that follows, centring is what keeps the refinement
      window on the true R peak.

    Neither changes the detection logic, but both change sample-level timing
    against a strict reading of the equations, so a reviewer comparing R-peak
    indices with a causal implementation should expect an offset there and
    not here.
    """
    from scipy.signal import butter, find_peaks, sosfiltfilt

    x = np.asarray(x, dtype=float).ravel()
    if fs <= 0:
        raise ValueError(f"`fs` must be positive, got {fs}.")
    nyq = 0.5 * fs
    # sosfiltfilt needs more samples than its padlen, and a QRS detector needs
    # enough signal to hold a beat regardless. Without this the failure is
    # scipy's "The length of the input vector x must be greater than padlen",
    # which says nothing about ECG. One second is the floor: the 150 ms
    # integration window and 200 ms refractory period are meaningless below it.
    min_samples = max(int(round(fs)), 40)
    if x.size < min_samples:
        raise ValueError(
            f"need at least {min_samples} samples (1 s at fs={fs:g} Hz) to "
            f"detect QRS complexes, got {x.size}."
        )
    # 1) bandpass 5–15 Hz
    sos = butter(3, [5.0 / nyq, min(15.0, nyq * 0.95) / nyq], btype="band", output="sos")
    bp = sosfiltfilt(sos, x)
    # 2) differentiate (Pan-Tompkins coefficients)
    der = np.zeros_like(bp)
    for n in range(4, bp.size):
        der[n] = (1.0 / 8.0) * (2 * bp[n] + bp[n - 1] - bp[n - 3] - 2 * bp[n - 4])
    # 3) square
    sq = der**2
    # 4) moving-window integration over 150 ms
    W = max(1, int(round(0.150 * fs)))
    kernel = np.ones(W) / W
    integ = np.convolve(sq, kernel, mode="same")
    # 5) detect peaks
    refractory = int(round(0.200 * fs))
    thr = 0.30 * integ.max() if integ.max() > 0 else 0.0
    peaks, _ = find_peaks(integ, height=thr, distance=max(1, refractory))
    # Optional: refine each peak to local max of bandpass-filtered signal
    half = int(round(0.05 * fs))
    refined = []
    for p in peaks:
        lo = max(0, p - half)
        hi = min(x.size, p + half + 1)
        refined.append(lo + int(np.argmax(np.abs(bp[lo:hi]))))
    r_peaks = np.asarray(refined, dtype=int)
    rr_ms = np.diff(r_peaks) * (1000.0 / fs) if r_peaks.size > 1 else np.array([])
    hr = float(60000.0 / rr_ms.mean()) if rr_ms.size else float("nan")
    res = RichResult(
        title="QRS detection (Pan-Tompkins)",
        summary_lines=[
            ("Fs (Hz)", float(fs)),
            ("R-peaks", int(r_peaks.size)),
            ("Mean HR (bpm)", hr),
            ("Threshold", float(thr)),
        ],
        interpretation=(f"Detected {r_peaks.size} R-peaks; mean HR {hr:.1f} bpm."),
        payload={
            "r_peaks": r_peaks,
            "rr_intervals_ms": rr_ms,
            "heart_rate_bpm": hr,
            "integrated": integ,
            "fs": float(fs),
        },
    )
    return with_describe_pointer(res, "rgqrs")


# CANONICAL TEST
# >>> fs = 360.0
# >>> t = np.arange(int(5*fs))/fs
# >>> # Simulated ECG-like impulse train at 1 Hz with Gaussian spikes
# >>> sig = np.zeros_like(t)
# >>> for tk in np.arange(0.5, 5.0, 1.0):
# ...     sig += np.exp(-((t - tk)*30)**2)
# >>> r = rangayyan_qrs_detect(sig, fs=fs)
# >>> 3 <= r["r_peaks"].size <= 6
# True


def cheatsheet():
    return "rgqrs: Pan-Tompkins QRS detector -- Rangayyan Ch 6"
