# morie.fn -- function file (rootcoder007/morie)
"""PSD to autocorrelation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rangayyan_psd_to_acf"]


def rangayyan_psd_to_acf(psd, freqs=None):
    r"""Autocorrelation from the power spectral density (Rangayyan
    Ch. 3):

    .. math:: R_{xx}(m) = \mathrm{IDFT}\{S_{xx}(f)\},

    the Wiener-Khinchin relation. Because the PSD supplied is
    one-sided (rfft convention), the inverse uses ``irfft`` so the
    result is real by construction -- taking a complex ifft and
    discarding the imaginary part would silently hide an asymmetric
    input.

    Parameters
    ----------
    psd : array-like
        One-sided power spectral density.
    freqs : array-like, optional
        Frequency grid, used only to report the lag spacing.

    Returns
    -------
    RichResult
        keys: ``acf``, ``lags``, ``r0`` (total power), ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (Wiener-Khinchin).
    """
    S = np.asarray(psd, dtype=float).ravel()
    if S.size < 2:
        raise ValueError("psd must have at least 2 points.")
    if np.any(S < 0):
        raise ValueError("a power spectral density cannot be negative.")
    acf = np.fft.irfft(S)
    n_lag = acf.size // 2 + 1
    lags = np.arange(n_lag)
    if freqs is not None:
        f = np.asarray(freqs, dtype=float).ravel()
        if f.size != S.size:
            raise ValueError("freqs must match the length of psd.")
    return RichResult(payload={"acf": acf[:n_lag], "lags": lags,
                               "r0": float(acf[0]),
                               "method": "R_xx = irfft(S_xx); real by construction"})


def cheatsheet():
    return "rgpsdacf: Wiener-Khinchin via irfft, so the result is real by construction"
