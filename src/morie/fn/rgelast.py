# morie.fn -- function file (rootcoder007/morie)
"""Heart-sound spectral stiffness index."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rangayyan_heart_elasticity"]


from .rgwelch import rangayyan_welch_psd


def rangayyan_heart_elasticity(pcg, fs, s1_window=None):
    r"""Spectral index of myocardial stiffness from S1 (Rangayyan
    Ch. 3).

    Higher stiffness shifts the S1 spectrum upward, so the dominant
    frequency and spectral centroid of the first heart sound track
    elasticity. This returns those descriptors -- it does NOT return a
    stiffness value in physical units: the relationship is monotone
    but the calibration is subject- and instrument-specific, and
    inventing an absolute number would be a fabrication.

    Parameters
    ----------
    pcg : array-like
        Phonocardiogram, or an isolated S1 segment.
    fs : float
        Sampling frequency.
    s1_window : (int, int), optional
        Sample range holding S1; the whole record if omitted.

    Returns
    -------
    RichResult
        keys: ``dominant_frequency``, ``spectral_centroid``,
        ``bandwidth_3db``, ``freqs``, ``psd``, ``calibrated`` (False),
        ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (heart sounds; spectral indices).
    """
    x = np.asarray(pcg, dtype=float).ravel()
    fs = float(fs)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}.")
    if s1_window is not None:
        a, b = int(s1_window[0]), int(s1_window[1])
        if not 0 <= a < b <= x.size:
            raise ValueError(f"s1_window ({a}, {b}) is out of range.")
        x = x[a:b]
    if x.size < 16:
        raise ValueError(f"need at least 16 samples, got {x.size}.")
    w = rangayyan_welch_psd(x, fs=fs, nperseg=min(256, x.size))
    f, p = w["freqs"], w["psd"]
    tot = float(p.sum())
    centroid = float(np.sum(f * p) / tot) if tot > 0 else np.nan
    ipk = int(np.argmax(p))
    above = np.flatnonzero(p >= p[ipk] / 2.0)
    bw = float(f[above[-1]] - f[above[0]]) if above.size else 0.0
    return RichResult(payload={"dominant_frequency": float(f[ipk]),
                               "spectral_centroid": centroid, "bandwidth_3db": bw,
                               "freqs": f, "psd": p, "calibrated": False,
                               "method": "S1 spectral descriptors; monotone in stiffness, NOT calibrated"})


def cheatsheet():
    return "rgelast: returns spectral descriptors, not an invented stiffness value"
