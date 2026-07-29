# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Haar wavelet universal-threshold smoothing."""

import math

import numpy as np

from ._richresult import RichResult

__all__ = ["wasserman_wavelet_smooth"]


def _haar_forward(x):
    coeffs = []
    a = x.copy()
    while a.size > 1:
        even, odd = a[0::2], a[1::2]
        coeffs.append((even - odd) / math.sqrt(2.0))
        a = (even + odd) / math.sqrt(2.0)
    return a, coeffs


def _haar_inverse(a, coeffs):
    for d in reversed(coeffs):
        even = (a + d) / math.sqrt(2.0)
        odd = (a - d) / math.sqrt(2.0)
        out = np.empty(even.size * 2)
        out[0::2], out[1::2] = even, odd
        a = out
    return a


def wasserman_wavelet_smooth(y, wavelet="haar", sigma=None):
    """
    Wavelet denoising with the universal threshold.

    Formula: lambda = sigma sqrt(2 log n); detail coefficients are
    HARD-thresholded (kept iff |d| > lambda), the approximation is
    untouched, and the signal is reconstructed. Only the Haar
    wavelet is implemented — any other name raises rather than
    silently substituting. sigma = None estimates the noise level by
    the MAD of the finest-scale details / 0.6744897501960817.

    Parameters
    ----------
    y : array-like
        Signal, length a power of 2, >= 2.
    wavelet : str
        Only "haar".
    sigma : float, optional
        Noise sd; None = MAD estimate.

    Returns
    -------
    result : dict
        Keys: estimate (denoised signal), threshold, sigma_used,
        n_kept, n_detail, n, method.

    References
    ----------
    Wasserman (2004), Ch 20 (wavelets); Donoho & Johnstone (1994).

    Examples
    --------
    A constant signal survives untouched; pure tiny noise dies:

    >>> out = wasserman_wavelet_smooth([3.0, 3.0, 3.0, 3.0], sigma=1.0)
    >>> [round(v, 10) for v in out["estimate"]]
    [3.0, 3.0, 3.0, 3.0]
    >>> out["n_kept"]
    0
    >>> big = wasserman_wavelet_smooth([10.0, -10.0, 10.0, -10.0], sigma=1.0)
    >>> big["n_kept"]
    2
    >>> [round(v, 10) for v in big["estimate"]]
    [10.0, -10.0, 10.0, -10.0]
    >>> wasserman_wavelet_smooth([1.0, 2.0, 3.0], sigma=1.0)
    Traceback (most recent call last):
        ...
    ValueError: the Haar transform needs a power-of-2 length; got 3.
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = y.size
    if wavelet != "haar":
        raise ValueError(f"only the Haar wavelet is implemented; got '{wavelet}'.")
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError(f"the Haar transform needs a power-of-2 length; got {n}.")
    approx, coeffs = _haar_forward(y)
    if sigma is None:
        finest = coeffs[0]
        mad = float(np.median(np.abs(finest - np.median(finest))))
        sigma = mad / 0.6744897501960817
        if sigma == 0:
            sigma = 0.0
    sigma = float(sigma)
    lam = sigma * math.sqrt(2.0 * math.log(n))
    kept = 0
    total = 0
    thr = []
    for d in coeffs:
        keep = np.abs(d) > lam
        kept += int(np.sum(keep))
        total += int(d.size)
        thr.append(np.where(keep, d, 0.0))
    rec = _haar_inverse(approx, thr)
    return RichResult(payload={
        "estimate": [float(v) for v in rec], "threshold": float(lam),
        "sigma_used": sigma, "n_kept": kept, "n_detail": total,
        "n": int(n),
        "method": "Haar + hard universal threshold sigma sqrt(2 log n)"})


def cheatsheet():
    return "wsmwlt: Haar cascade, hard threshold lam = sigma sqrt(2 log n); MAD sigma"
