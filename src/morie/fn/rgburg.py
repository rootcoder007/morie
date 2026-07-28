# morie.fn -- function file (rootcoder007/morie)
"""Burg AR estimation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["rangayyan_burg_method"]


def rangayyan_burg_method(x, order=8, fs=1.0):
    r"""Burg's lattice method for AR estimation (Rangayyan Ch. 3).

    Recursively chooses each reflection coefficient to minimise the
    sum of FORWARD and BACKWARD prediction error powers:

    .. math:: k_m = \frac{-2\sum_n f_{m-1}(n) b_{m-1}(n-1)}
              {\sum_n f_{m-1}^2(n) + \sum_n b_{m-1}^2(n-1)}.

    Unlike Yule-Walker it never forms an autocorrelation estimate, so
    it needs no windowing and gives better resolution on short
    records; and because every :math:`|k_m| \le 1` by construction,
    the resulting model is guaranteed stable.

    Parameters
    ----------
    x : array-like
        Signal.
    order : int, default 8
        AR order.
    fs : float, default 1.0
        Sampling frequency, carried through for spectra.

    Returns
    -------
    RichResult
        keys: ``a``, ``reflection``, ``sigma2``, ``order``,
        ``stable``, ``fs``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (Burg's method).
    """
    x = np.asarray(x, dtype=float).ravel()
    p = int(order)
    N = x.size
    if p < 1:
        raise ValueError(f"order must be at least 1, got {p}.")
    if N < p + 1:
        raise ValueError(f"need more than order = {p} samples, got {N}.")
    if float(fs) <= 0:
        raise ValueError("fs must be positive.")
    f = x.copy()
    b = x.copy()
    a = np.zeros(0)
    E = float(np.mean(x**2))
    ks = []
    for m in range(1, p + 1):
        fm = f[m:]
        bm = b[m - 1 : -1]
        den = float(np.dot(fm, fm) + np.dot(bm, bm))
        k = -2.0 * float(np.dot(fm, bm)) / den if den > 0 else 0.0
        k = float(np.clip(k, -1.0, 1.0))  # |k| <= 1 keeps the model stable
        ks.append(k)
        a_new = np.r_[a, 0.0] + k * np.r_[a[::-1], 1.0] if a.size else np.array([k])
        a = a_new
        f_new = fm + k * bm
        b_new = bm + k * fm
        f = np.r_[np.zeros(m), f_new]
        b = np.r_[np.zeros(m), b_new]
        E *= 1.0 - k**2
    roots = np.roots(np.r_[1.0, a])
    return RichResult(payload={"a": a, "reflection": np.array(ks), "sigma2": float(E),
                               "order": p, "stable": bool(np.all(np.abs(roots) < 1.0)),
                               "fs": float(fs),
                               "method": "Burg lattice; |k| <= 1 guarantees a stable model"})


def cheatsheet():
    return "rgburg: no ACF, no windowing; |k|<=1 makes stability automatic"
