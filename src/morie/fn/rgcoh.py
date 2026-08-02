# morie.fn -- function file (rootcoder007/morie)
"""Magnitude-squared coherence -- Rangayyan & Krishnan Sec 4.5.1."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_coherence"]


def rangayyan_coherence(x, y, fs=1.0, nperseg=None):
    """Magnitude-squared coherence::

        C_xy(f) = |S_xy(f)|² / (S_xx(f) S_yy(f))

    Welch cross/auto-spectra; returns one-sided coherence in [0, 1].

    Parameters
    ----------
    x, y : array-like
    fs : float
    nperseg : int, optional

    Returns
    -------
    RichResult with keys ``freqs``, ``coherence``, ``mean_coherence``,
    ``peak_freq``, ``peak_coherence``.

    References
    ----------
    Rangayyan, R. M., & Krishnan, S. (2024). *Biomedical Signal Analysis*
        (3rd ed.). Wiley-IEEE Press. Sec 4.5.1 "Coherence analysis of EEG
        channels", p.235.
    """
    from ._signal_core import coherence

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length.")
    if nperseg is None:
        nperseg = min(x.size, 256)
    f, Cxy = coherence(x, y, fs=fs, nperseg=nperseg)
    peak = int(np.argmax(Cxy))
    res = RichResult(
        title="Magnitude-squared coherence",
        summary_lines=[
            ("Fs (Hz)", float(fs)),
            ("nperseg", int(nperseg)),
            ("Mean coherence", float(Cxy.mean())),
            ("Peak coherence", float(Cxy[peak])),
            ("Peak freq (Hz)", float(f[peak])),
        ],
        interpretation=f"Peak coherence {Cxy[peak]:.3g} at {f[peak]:.3g} Hz.",
        payload={
            "freqs": f,
            "coherence": Cxy,
            "mean_coherence": float(Cxy.mean()),
            "peak_freq": float(f[peak]),
            "peak_coherence": float(Cxy[peak]),
        },
    )
    return with_describe_pointer(res, "rgcoh")


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> fs = 100.0; t = np.arange(1024)/fs
# >>> a = np.sin(2*np.pi*10*t)
# >>> b = a + 0.1*rng.standard_normal(t.size)
# >>> r = rangayyan_coherence(a, b, fs=fs)
# >>> r["peak_coherence"] > 0.5
# True


def cheatsheet():
    return "rgcoh: magnitude-squared coherence -- Rangayyan & Krishnan Sec 4.5.1"
