# morie.fn -- function file (rootcoder007/morie)
"""Cepstral pitch detection."""

import numpy as np

from ._richresult import RichResult

__all__ = ["rangayyan_cepstrum_pitch"]


def rangayyan_cepstrum_pitch(x, fs, f0_range=(50.0, 500.0)):
    r"""Pitch from the real cepstrum (Rangayyan Ch. 3):

    .. math:: c(q) = \mathrm{IDFT}\{\log |X(f)|\},

    with the pitch period :math:`T_0` at the quefrency of the
    dominant rahmonic. The logarithm is what makes this work: it turns
    the product of excitation and vocal-tract spectra into a SUM, so
    the periodic excitation separates from the smooth envelope in
    quefrency. Searching only inside ``f0_range`` avoids locking onto
    the low-quefrency envelope peak.

    Parameters
    ----------
    x : array-like
        Signal.
    fs : float
        Sampling frequency.
    f0_range : (float, float)
        Plausible pitch range in Hz.

    Returns
    -------
    RichResult
        keys: ``f0``, ``period_s``, ``quefrency``, ``cepstrum``,
        ``peak_value``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (cepstral analysis).
    """
    x = np.asarray(x, dtype=float).ravel()
    fs = float(fs)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}.")
    lo, hi = float(f0_range[0]), float(f0_range[1])
    if not 0 < lo < hi:
        raise ValueError(f"f0_range must satisfy 0 < lo < hi, got {f0_range}.")
    if x.size < 16:
        raise ValueError(f"need at least 16 samples, got {x.size}.")
    spec = np.abs(np.fft.rfft(x))
    ceps = np.fft.irfft(np.log(np.maximum(spec, 1e-300)))
    q = np.arange(ceps.size) / fs
    q_lo, q_hi = 1.0 / hi, 1.0 / lo
    band = np.flatnonzero((q >= q_lo) & (q <= q_hi))
    if band.size == 0:
        raise ValueError("f0_range maps outside the available quefrencies.")
    ipk = band[int(np.argmax(ceps[band]))]
    T0 = float(q[ipk])
    return RichResult(payload={"f0": 1.0 / T0 if T0 > 0 else np.nan, "period_s": T0,
                               "quefrency": ipk, "cepstrum": ceps,
                               "peak_value": float(ceps[ipk]),
                               "method": "log turns convolution into addition; search inside f0_range"})


def cheatsheet():
    return "rgcepsp: the log separates excitation from envelope in quefrency"
