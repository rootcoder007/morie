# morie.fn -- function file (rootcoder007/morie)
"""EEG band power (delta theta alpha beta gamma) -- Rangayyan & Krishnan Sec 4.4.1."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_eeg_bands"]

# np.trapz was REMOVED in NumPy 2.0 and renamed to np.trapezoid. pyproject
# still declares numpy>=1.24, where only np.trapz exists, so bind whichever
# the installed NumPy provides. Without this the function raised
# AttributeError on every call under NumPy >= 2 -- not a test artefact, a
# hard failure for any caller.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz



_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}


def rangayyan_eeg_bands(x, fs, bands=None, nperseg=None):
    """Absolute and relative band power in canonical EEG bands.

    P_band = ∫ S(f) df via Welch's PSD over band (lo, hi].

    Parameters
    ----------
    x : array-like
    fs : float
    bands : dict, optional
    nperseg : int, optional

    Returns
    -------
    RichResult with keys ``absolute`` (band->W), ``relative`` (band->fraction),
    ``total_power``, ``freqs``, ``psd``.

    References
    ----------
    Rangayyan, R. M., & Krishnan, S. (2024). *Biomedical Signal Analysis*
        (3rd ed.). Wiley-IEEE Press. Sec 4.4.1 "Detection of EEG rhythms",
        p.228. The previous docstring cited Ch 9.
    """
    from scipy.signal import welch

    x = np.asarray(x, dtype=float)
    if nperseg is None:
        nperseg = max(16, min(x.size, int(4 * fs)))
    bands = bands or _BANDS
    freqs, pxx = welch(x, fs=fs, nperseg=nperseg)
    df = float(freqs[1] - freqs[0])
    total = float(_trapezoid(pxx, freqs))
    absolute = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        absolute[name] = float(_trapezoid(pxx[mask], freqs[mask])) if mask.any() else 0.0
    relative = {k: (v / total if total > 0 else 0.0) for k, v in absolute.items()}
    rows = [[name, f"{absolute[name]:.4g}", f"{relative[name] * 100:.2f}%"] for name in bands]
    res = RichResult(
        title="EEG band power",
        summary_lines=[("Fs (Hz)", float(fs)), ("Total power", total), ("Bin width (Hz)", df)],
        tables=[{"title": "Power by band", "headers": ["Band", "Absolute (W)", "Relative"], "rows": rows}],
        interpretation="Relative percentages sum ≤100% (residual outside defined bands).",
        payload={"absolute": absolute, "relative": relative, "total_power": total, "freqs": freqs, "psd": pxx},
    )
    return with_describe_pointer(res, "rgeeg")


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> fs = 256.0
# >>> t = np.arange(2048)/fs
# >>> x = np.sin(2*np.pi*10*t) + 0.3*rng.standard_normal(t.size)
# >>> r = rangayyan_eeg_bands(x, fs=fs)
# >>> r["absolute"]["alpha"] > r["absolute"]["gamma"]
# True


def cheatsheet():
    return "rgeeg: EEG delta theta alpha beta gamma band power -- Rangayyan & Krishnan Sec 4.4.1"
