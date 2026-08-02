# morie.fn -- function file (rootcoder007/morie)
"""Envelogram."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rangayyan_envelogram"]


def rangayyan_envelogram(pcg, ecg=None, fs=1000.0, r_peaks=None, n_beats=None):
    r"""Envelogram of a PCG signal (Rangayyan Ch. 3):

    .. math:: \mathrm{env}_{avg}[n] = \frac1M \sum_{k=1}^{M}
              \big| x_k(n) + j\,\mathcal H\{x_k(n)\} \big|,

    the ensemble-averaged analytic-signal magnitude. The Hilbert
    transform gives the instantaneous amplitude envelope, which is
    what makes S1 and S2 visible as smooth bumps rather than as
    oscillation. Alignment comes from the ECG R peaks -- averaging
    unaligned beats smears the envelope and is the usual way this
    goes wrong, so the R peaks are required rather than guessed.

    Parameters
    ----------
    pcg : array-like
        Phonocardiogram.
    ecg : array-like, optional
        ECG used only if r_peaks must be detected.
    fs : float, default 1000.0
        Sampling frequency.
    r_peaks : array-like of int, optional
        R-peak sample indices; detected from ecg when omitted.
    n_beats : int, optional
        Beat-count check.

    Returns
    -------
    RichResult
        keys: ``envelope`` (averaged), ``beats`` (M, L matrix),
        ``M``, ``beat_length``, ``fs``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (envelope extraction, envelograms).
    """
    from scipy import signal as sig

    x = np.asarray(pcg, dtype=float).ravel()
    fs = float(fs)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}.")
    if r_peaks is None:
        if ecg is None:
            raise ValueError(
                "supply r_peaks, or an ecg to detect them from -- averaging "
                "unaligned beats smears the envelope."
            )
        e = np.asarray(ecg, dtype=float).ravel()
        thr = float(np.mean(e) + 2.0 * np.std(e))
        pk, _ = sig.find_peaks(e, height=thr, distance=int(0.25 * fs))
        r_peaks = pk
    r = np.asarray(r_peaks, dtype=int).ravel()
    if r.size < 2:
        raise ValueError("need at least 2 R peaks to segment beats.")
    if n_beats is not None and int(n_beats) != r.size:
        raise ValueError(f"n_beats = {n_beats} does not match {r.size} peaks.")
    L = int(np.min(np.diff(r)))
    if L < 8:
        raise ValueError("beats are too short to average.")
    env = np.abs(sig.hilbert(x))
    beats = np.array([env[p : p + L] for p in r[:-1] if p + L <= env.size])
    if beats.size == 0:
        raise ValueError("no complete beats within the signal.")
    return RichResult(payload={"envelope": beats.mean(axis=0), "beats": beats,
                               "M": int(beats.shape[0]), "beat_length": L, "fs": fs,
                               "method": "Hilbert envelope averaged over R-aligned beats"})


def cheatsheet():
    return "rgenvgm: R-peak alignment is required -- unaligned averaging smears the envelope"
