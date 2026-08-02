# morie.fn -- function file (rootcoder007/morie)
"""Power spectral density via Welch's method -- Rangayyan & Krishnan Sec 6.3.2-6.3.4."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_psd"]

# np.trapz was REMOVED in NumPy 2.0 and renamed to np.trapezoid. pyproject
# still declares numpy>=1.24, where only np.trapz exists, so bind whichever
# the installed NumPy provides. Without this the function raised
# AttributeError on every call under NumPy >= 2 -- not a test artefact, a
# hard failure for any caller.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz



def rangayyan_psd(x, fs=1.0, nperseg=None, noverlap=None, window="hann"):
    """Welch periodogram PSD.

    Computes the one-sided PSD using overlapping windowed segments::

        S(f) = (1 / (K * W * fs)) * sum_k |X_k(f)|^2

    where ``W`` is the window's noise-equivalent bandwidth scaling.

    Parameters
    ----------
    x : array-like
        Input signal.
    fs : float
        Sampling frequency (Hz).
    nperseg : int, optional
        Segment length. Default ``min(len(x), 256)``.
    noverlap : int, optional
        Overlap samples. Default ``nperseg//2``.
    window : str
        Window name (``hann`` default).

    Returns
    -------
    RichResult with keys ``freqs``, ``psd``, ``fs``, ``nperseg``, ``peak_freq``.

    References
    ----------
    Rangayyan, R. M., & Krishnan, S. (2024). *Biomedical Signal Analysis*
        (3rd ed.). Wiley-IEEE Press. Sec 6.3.2 "The periodogram", p.323;
        Sec 6.3.3 "The need for averaging PSDs", p.325; Sec 6.3.4 "The use
        of windows: spectral resolution and leakage", p.326 -- NOT Ch 4.
        Welch's method is precisely the averaged, windowed periodogram those
        three sections build up.
    Welch, P. D. (1967). The use of Fast Fourier Transform for the estimation
        of power spectra. *IEEE Transactions on Audio and Electroacoustics*,
        15(2), 70-73.
    """
    from ._signal_core import welch

    x = np.asarray(x, dtype=float)
    if nperseg is None:
        nperseg = min(x.size, 256)
    freqs, pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window, scaling="density")
    peak_idx = int(np.argmax(pxx))
    res = RichResult(
        title="Power spectral density (Welch)",
        summary_lines=[
            ("Fs (Hz)", float(fs)),
            ("nperseg", int(nperseg)),
            ("Window", window),
            ("Peak freq (Hz)", float(freqs[peak_idx])),
            ("Peak power", float(pxx[peak_idx])),
            ("Total power", float(_trapezoid(pxx, freqs))),
        ],
        interpretation=(f"PSD peaks at {freqs[peak_idx]:.3g} Hz; total band power {float(_trapezoid(pxx, freqs)):.4g}."),
        payload={
            "freqs": freqs,
            "psd": pxx,
            "fs": float(fs),
            "nperseg": int(nperseg),
            "peak_freq": float(freqs[peak_idx]),
            "total_power": float(_trapezoid(pxx, freqs)),
        },
    )
    return with_describe_pointer(res, "rgpsd")


# CANONICAL TEST
# >>> fs=100.0; t=np.arange(1000)/fs
# >>> x = np.sin(2*np.pi*10*t)
# >>> r = rangayyan_psd(x, fs=fs, nperseg=256)
# >>> abs(r["peak_freq"] - 10.0) < 1.0
# True


def cheatsheet():
    return "rgpsd: Welch power spectral density -- Rangayyan & Krishnan Sec 6.3.3"
