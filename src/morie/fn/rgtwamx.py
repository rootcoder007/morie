# morie.fn -- function file (rootcoder007/morie)
"""T-wave alternans spectral method."""

import numpy as np

from ._richresult import RichResult

__all__ = ["rangayyan_twa_spectral_mx"]


def rangayyan_twa_spectral_mx(ecg, fs, r_peaks, t_window=None, n_beats=128):
    r"""Spectral T-wave alternans (Rangayyan Ch. 3).

    Builds a beat-by-sample matrix aligned on the R peaks, takes the
    FFT ALONG THE BEAT AXIS at each sample offset, and reads the power
    at 0.5 cycles/beat -- the frequency of a strictly ABAB alternation:

    .. math:: k_{alt} = \tfrac12 \text{ cycles per beat}.

    The alternans voltage is the excess over the neighbouring noise
    band, and the k-score is that excess in noise standard deviations;
    both are returned because a raw spectral peak means nothing
    without its noise floor. An even number of beats is required, or
    0.5 cycles/beat is not an exact FFT bin and the alternans power
    leaks.

    Parameters
    ----------
    ecg : array-like
        ECG signal.
    fs : float
        Sampling frequency.
    r_peaks : array-like of int
        R-peak indices.
    t_window : (int, int), optional
        Offsets after R defining the T wave; a physiological default
        of 100-300 ms is used otherwise.
    n_beats : int, default 128
        Beats to use (truncated to an even number).

    Returns
    -------
    RichResult
        keys: ``alternans_voltage``, ``k_score``, ``noise_mean``,
        ``noise_std``, ``spectrum``, ``n_beats_used``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (T-wave alternans).
    """
    x = np.asarray(ecg, dtype=float).ravel()
    fs = float(fs)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}.")
    r = np.asarray(r_peaks, dtype=int).ravel()
    if r.size < 8:
        raise ValueError(f"need at least 8 beats, got {r.size}.")
    lo, hi = (int(0.10 * fs), int(0.30 * fs)) if t_window is None else (
        int(t_window[0]), int(t_window[1])
    )
    if not 0 <= lo < hi:
        raise ValueError("t_window must satisfy 0 <= start < stop.")
    usable = [p for p in r if p + hi <= x.size]
    M = min(int(n_beats), len(usable))
    M -= M % 2  # 0.5 cycles/beat must land on an exact FFT bin
    if M < 8:
        raise ValueError("fewer than 8 complete beats after alignment.")
    mat = np.array([x[p + lo : p + hi] for p in usable[:M]])
    mat = mat - mat.mean(axis=0, keepdims=True)
    S = np.abs(np.fft.rfft(mat, axis=0)) ** 2 / M
    spec = S.sum(axis=1)  # aggregate across the T-wave samples
    k_alt = M // 2  # the 0.5 cycles/beat bin
    noise_band = spec[int(0.33 * len(spec)) : k_alt]
    nm = float(noise_band.mean()) if noise_band.size else 0.0
    ns = float(noise_band.std()) if noise_band.size else 0.0
    excess = float(spec[k_alt]) - nm
    volt = float(np.sqrt(max(excess, 0.0)))
    return RichResult(payload={"alternans_voltage": volt,
                               "k_score": (excess / ns) if ns > 0 else np.inf,
                               "noise_mean": nm, "noise_std": ns, "spectrum": spec,
                               "n_beats_used": int(M),
                               "method": "FFT along the beat axis; power at 0.5 cyc/beat over noise"})


def cheatsheet():
    return "rgtwamx: even beat count required or 0.5 cyc/beat is not an exact bin"
