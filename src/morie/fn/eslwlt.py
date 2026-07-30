# morie.fn -- function file (rootcoder007/morie)
"""Wavelet smoothing -- ESL Sec 5.9."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["esl_wavelet_smooth"]


def esl_wavelet_smooth(y, wavelet="haar", mode="soft", threshold=None, levels=None):
    r"""Denoise a signal by thresholding its discrete wavelet transform.

    Take the DWT, shrink the detail coefficients toward zero, invert. With
    the universal threshold of Donoho & Johnstone,

    .. math::
        \lambda = \hat\sigma \sqrt{2 \log n}, \qquad
        \hat\sigma = \operatorname{MAD}(d_1) / 0.6745,

    where :math:`d_1` are the finest-scale details -- almost pure noise for a
    signal with any smoothness, which is what makes the MAD a usable noise
    estimate at all.

    :math:`\sqrt{2\log n}` is chosen so that with high probability *no* pure
    noise coefficient survives; the shrinkage is therefore deliberately
    conservative, and a smoother-than-expected fit is the method working as
    designed rather than over-smoothing.

    Soft thresholding subtracts :math:`\lambda` from every surviving
    coefficient, so it is biased toward zero but continuous. Hard keeps
    coefficients intact above :math:`\lambda` -- unbiased, but discontinuous
    in the data, which shows up as spurious blips.

    Implemented with the Haar transform natively so no wavelet library is
    required; ``wavelet="haar"`` is the only supported basis.

    Parameters
    ----------
    y : array-like
        Signal. Length is padded to a power of two by symmetric reflection.
    wavelet : {"haar"}
        Wavelet basis.
    mode : {"soft", "hard"}
        Shrinkage rule.
    threshold : float, optional
        Explicit :math:`\lambda`. Defaults to the universal threshold.
    levels : int, optional
        Decomposition depth. Defaults to the maximum.

    Returns
    -------
    RichResult
        ``signal`` (denoised, original length), ``threshold``, ``sigma``,
        ``coefficients``, ``n_zeroed``, ``levels``.

    References
    ----------
    Donoho, D. L., & Johnstone, I. M. (1994). Ideal spatial adaptation by
        wavelet shrinkage. *Biometrika*, 81(3), 425-455.
    Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
        Statistical Learning* (2nd ed.). Springer.

    Examples
    --------
    Denoising a blocky signal reduces the error against the truth.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> clean = np.repeat([0.0, 4.0, 1.0, -2.0], 64)
    >>> noisy = clean + rng.normal(0, 0.5, 256)
    >>> r = esl_wavelet_smooth(noisy)
    >>> bool(np.mean((r["signal"] - clean) ** 2) < np.mean((noisy - clean) ** 2))
    True

    The MAD-based noise estimate recovers the injected sigma.

    >>> bool(abs(r["sigma"] - 0.5) < 0.15)
    True

    Soft thresholding cannot increase energy, since every coefficient moves
    toward zero.

    >>> pure = rng.normal(size=256)
    >>> s = esl_wavelet_smooth(pure, mode="soft")["signal"]
    >>> bool(np.sum(s ** 2) <= np.sum(pure ** 2) + 1e-9)
    True

    >>> esl_wavelet_smooth(noisy, wavelet="db4")
    Traceback (most recent call last):
        ...
    ValueError: only the Haar basis is implemented; got 'db4'
    """
    if wavelet != "haar":
        raise ValueError(f"only the Haar basis is implemented; got {wavelet!r}")
    if mode not in ("soft", "hard"):
        raise ValueError('mode must be "soft" or "hard"')
    y = np.asarray(y, dtype=float).ravel()
    n0 = y.size
    if n0 < 2:
        raise ValueError("need at least 2 observations")

    n = 1 << int(np.ceil(np.log2(n0)))
    pad = np.r_[y, y[::-1]][:n] if n > n0 else y.copy()

    max_lev = int(np.log2(n))
    levels = max_lev if levels is None else min(int(levels), max_lev)

    approx = pad.copy()
    details = []
    for _ in range(levels):
        even, odd = approx[0::2], approx[1::2]
        approx = (even + odd) / np.sqrt(2.0)
        details.append((even - odd) / np.sqrt(2.0))

    d1 = details[0]
    sigma = float(np.median(np.abs(d1 - np.median(d1))) / 0.6745)
    lam = float(sigma * np.sqrt(2.0 * np.log(n))) if threshold is None else float(threshold)

    zeroed = 0
    shrunk = []
    for d in details:
        if mode == "soft":
            t = np.sign(d) * np.maximum(np.abs(d) - lam, 0.0)
        else:
            t = np.where(np.abs(d) > lam, d, 0.0)
        zeroed += int(np.sum(t == 0))
        shrunk.append(t)

    rec = approx
    for d in reversed(shrunk):
        even = (rec + d) / np.sqrt(2.0)
        odd = (rec - d) / np.sqrt(2.0)
        rec = np.empty(2 * rec.size)
        rec[0::2], rec[1::2] = even, odd

    return RichResult(
        title="Wavelet smoothing (Haar)",
        summary_lines=[("n", n0), ("levels", levels), ("sigma", sigma),
                       ("threshold", lam)],
        payload={
            "signal": rec[:n0], "threshold": lam, "sigma": sigma,
            "coefficients": shrunk, "approx": approx,
            "n_zeroed": zeroed, "levels": int(levels), "mode": mode,
            "method": "esl_wavelet_smooth",
        },
    )


def cheatsheet():
    return "eslwlt: Haar DWT + universal threshold sigma*sqrt(2 log n); deliberately conservative shrinkage"
