# morie.fn -- function file (rootcoder007/morie)
"""Periodogram."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rangayyan_periodogram"]


def rangayyan_periodogram(x, fs=1.0):
    r"""Periodogram power spectral density (Rangayyan Ch. 3):

    .. math:: P(f) = \frac1N |X(f)|^2, \qquad X(f) = \mathrm{DFT}(x).

    The periodogram is NOT a consistent estimator: its variance does
    not fall as N grows, only its frequency resolution improves. That
    is precisely why Welch's method (:mod:`morie.fn.rgwelch`) averages
    segments, and the returned docstring says so rather than presenting
    the periodogram as a finished estimate.

    Parameters
    ----------
    x : array-like
        Signal.
    fs : float, default 1.0
        Sampling frequency.

    Returns
    -------
    RichResult
        keys: ``freqs``, ``psd``, ``total_power``, ``N``, ``fs``,
        ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (the periodogram).
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    if N < 2:
        raise ValueError(f"need at least 2 samples, got {N}.")
    fs = float(fs)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}.")
    X = np.fft.rfft(x)
    psd = (np.abs(X) ** 2) / N
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    return RichResult(payload={"freqs": freqs, "psd": psd,
                               "total_power": float(np.mean(x**2)), "N": int(N),
                               "fs": fs,
                               "method": "P(f) = |DFT(x)|^2/N; inconsistent -- variance does not shrink"})


def cheatsheet():
    return "rgperio: inconsistent estimator; more N buys resolution, not precision"
