# morie.fn -- function file (rootcoder007/morie)
"""Pan-Tompkins peak classification."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch4_pan_tompkins_peak_classification"]


def rangayyan_ch4_pan_tompkins_peak_classification(PEAKI, SPKI=None, NPKI=None,
                                                   is_signal=None):
    r"""Pan-Tompkins adaptive threshold update (Rangayyan Ch. 4):

    .. math:: SPKI &= 0.125\,PEAKI + 0.875\,SPKI
              \quad\text{(signal peak)}\\
              NPKI &= 0.125\,PEAKI + 0.875\,NPKI
              \quad\text{(noise peak)}

    Two exponential trackers with the SAME 1/8 coefficient, updated
    according to which class the peak was assigned. The detection
    threshold is :math:`NPKI + 0.25(SPKI - NPKI)`, which floats
    between the two estimates so the detector adapts to changing
    amplitude without a fixed cutoff.

    Parameters
    ----------
    PEAKI : float or array-like
        Peak amplitude(s), processed in order.
    SPKI, NPKI : float, optional
        Running signal and noise estimates; initialised from the first
        peak when omitted.
    is_signal : bool or array-like of bool, optional
        Class of each peak; peaks above the current threshold are
        treated as signal when omitted.

    Returns
    -------
    RichResult
        keys: ``SPKI``, ``NPKI``, ``threshold``, ``classified``
        (per peak), ``n_peaks``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 4 (the Pan-Tompkins algorithm).
    """
    peaks = np.atleast_1d(np.asarray(PEAKI, dtype=float))
    if peaks.size < 1:
        raise ValueError("PEAKI must be non-empty.")
    if np.any(peaks < 0):
        raise ValueError("peak amplitudes must be non-negative.")
    spki = float(peaks[0]) if SPKI is None else float(SPKI)
    npki = float(peaks[0]) / 2.0 if NPKI is None else float(NPKI)
    flags = None
    if is_signal is not None:
        flags = np.atleast_1d(np.asarray(is_signal, dtype=bool))
        if flags.size != peaks.size:
            raise ValueError("is_signal must have one entry per peak.")
    classified = []
    for i, p in enumerate(peaks):
        thr = npki + 0.25 * (spki - npki)
        sig = bool(p > thr) if flags is None else bool(flags[i])
        if sig:
            spki = 0.125 * p + 0.875 * spki
        else:
            npki = 0.125 * p + 0.875 * npki
        classified.append(sig)
    return RichResult(payload={"SPKI": spki, "NPKI": npki,
                               "threshold": npki + 0.25 * (spki - npki),
                               "classified": np.array(classified),
                               "n_peaks": int(peaks.size),
                               "method": "Pan-Tompkins 1/8 trackers; threshold floats between them"})


def cheatsheet():
    return "rng190: threshold = NPKI + 0.25(SPKI-NPKI), adapts without a fixed cutoff"
