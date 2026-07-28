# morie.fn -- function file (rootcoder007/morie)
"""AR power spectrum."""

import numpy as np

from ._richresult import RichResult

__all__ = ["rangayyan_ar_spectrum"]


from .rgyw import rangayyan_yule_walker


def rangayyan_ar_spectrum(x, order=8, fs=1.0, n_freqs=512):
    r"""Autoregressive (parametric) power spectrum (Rangayyan Ch. 3):

    .. math:: S_{AR}(f) = \frac{\sigma^2}
              {\big|1 + \sum_k a_k e^{-j2\pi f k T}\big|^2}.

    Unlike the periodogram this is a smooth, all-pole spectrum with
    resolution not limited by the record length -- which is its appeal
    and its danger: choosing the order too high invents spectral peaks
    that are not in the data. The fitted model's stability is
    reported, since an unstable fit makes the spectrum meaningless.

    Parameters
    ----------
    x : array-like
        Signal.
    order : int, default 8
        AR order.
    fs : float, default 1.0
        Sampling frequency.
    n_freqs : int, default 512
        Frequency grid size.

    Returns
    -------
    RichResult
        keys: ``freqs``, ``psd``, ``a``, ``sigma2``, ``order``,
        ``stable``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (parametric spectral estimation).
    """
    fs = float(fs)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}.")
    n_freqs = int(n_freqs)
    if n_freqs < 8:
        raise ValueError(f"n_freqs must be at least 8, got {n_freqs}.")
    yw = rangayyan_yule_walker(x, order=order)
    a = yw["a"]
    freqs = np.linspace(0.0, fs / 2.0, n_freqs)
    k = np.arange(1, a.size + 1)
    expo = np.exp(-2j * np.pi * np.outer(freqs / fs, k))
    denom = np.abs(1.0 + expo @ a) ** 2
    psd = yw["sigma2"] / np.maximum(denom, 1e-300)
    return RichResult(payload={"freqs": freqs, "psd": psd, "a": a,
                               "sigma2": yw["sigma2"], "order": yw["order"],
                               "stable": yw["stable"],
                               "method": "All-pole AR spectrum; high order invents peaks"})


def cheatsheet():
    return "rgarsp: resolution not limited by record length -- but order too high fabricates peaks"
