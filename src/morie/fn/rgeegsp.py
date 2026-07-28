# morie.fn -- function file (rootcoder007/morie)
"""EEG band powers."""

import numpy as np

from ._richresult import RichResult

__all__ = ["rangayyan_eeg_spectral"]


from .rgwelch import rangayyan_welch_psd


def rangayyan_eeg_spectral(eeg, fs, n_ch=None):
    r"""EEG band powers (Rangayyan Ch. 3):

    .. math:: P_\delta = \int_0^4 S(f)\,df, \quad
              P_\theta = \int_4^8, \quad
              P_\alpha = \int_8^{13}, \quad
              P_\beta = \int_{13}^{30}.

    Band edges are those stated in the text. The PSD comes from
    Welch rather than a bare periodogram, since band POWER is an
    integral and integrating a high-variance estimate propagates that
    variance straight into the clinical number. Relative powers are
    returned alongside the absolute ones, because absolute EEG power
    depends on electrode impedance and is rarely comparable across
    recordings.

    Parameters
    ----------
    eeg : array-like, shape (N,) or (n_ch, N)
        Signal(s).
    fs : float
        Sampling frequency, must exceed 60 Hz to cover the beta band.
    n_ch : int, optional
        Channel count check.

    Returns
    -------
    RichResult
        keys: ``bands`` (dict of absolute power), ``relative``,
        ``total_power``, ``freqs``, ``psd``, ``n_ch``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (EEG spectral analysis).
    """
    X = np.atleast_2d(np.asarray(eeg, dtype=float))
    fs = float(fs)
    if fs <= 60.0:
        raise ValueError(
            f"fs = {fs} is too low: the 13-30 Hz beta band needs fs > 60 Hz."
        )
    m, N = X.shape
    if n_ch is not None and int(n_ch) != m:
        raise ValueError(f"n_ch = {n_ch} does not match the {m} channels.")
    edges = {"delta": (0.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
             "beta": (13.0, 30.0)}
    bands = {k: np.zeros(m) for k in edges}
    psd_all, freqs = [], None
    for c in range(m):
        w = rangayyan_welch_psd(X[c], fs=fs)
        freqs = w["freqs"]
        psd = w["psd"]
        psd_all.append(psd)
        for k, (a, b) in edges.items():
            sel = (freqs >= a) & (freqs < b)
            bands[k][c] = float(np.trapezoid(psd[sel], freqs[sel])) if sel.any() else 0.0
    total = sum(bands[k] for k in edges)
    rel = {k: np.where(total > 0, bands[k] / np.maximum(total, 1e-300), 0.0)
           for k in edges}
    squeeze = m == 1
    return RichResult(payload={
        "bands": {k: (float(v[0]) if squeeze else v) for k, v in bands.items()},
        "relative": {k: (float(v[0]) if squeeze else v) for k, v in rel.items()},
        "total_power": float(total[0]) if squeeze else total,
        "freqs": freqs, "psd": psd_all[0] if squeeze else np.array(psd_all),
        "n_ch": int(m),
        "method": "Welch PSD integrated over the book's band edges; relative powers too"})


def cheatsheet():
    return "rgeegsp: absolute EEG power is impedance-dependent -- use the relative values"
