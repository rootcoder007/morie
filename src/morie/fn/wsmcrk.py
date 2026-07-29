# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Nadaraya-Watson kernel regression."""

import numpy as np

from ._richresult import RichResult

__all__ = ["wasserman_kernel_regression"]


def wasserman_kernel_regression(x, x_data, y_data, h):
    """
    Nadaraya-Watson kernel regression with a Gaussian kernel.

    Formula: m_h(x) = sum_i K_h(x - X_i) Y_i / sum_i K_h(x - X_i),
    K the standard normal kernel. An evaluation point whose kernel
    weights all underflow to zero (far outside the data at small h)
    is reported as nan rather than fabricated.

    Parameters
    ----------
    x : array-like
        Evaluation point(s).
    x_data, y_data : array-like
        Paired training sample, equal length >= 1.
    h : float
        Bandwidth, > 0.

    Returns
    -------
    result : dict
        Keys: estimate (m at the first x), values, effective_n
        (kernel-weight sum at each x), h, n, method.

    References
    ----------
    Wasserman (2004), Ch 20, section 20.4 (Definition 20.21).

    Examples
    --------
    Huge bandwidth returns the global mean everywhere; symmetric
    data about the query point averages symmetrically:

    >>> out = wasserman_kernel_regression(0.0, [-1.0, 1.0], [2.0, 4.0], 1e6)
    >>> round(out["estimate"], 12)
    3.0
    >>> out2 = wasserman_kernel_regression(1.0, [-1.0, 1.0], [2.0, 4.0], 0.1)
    >>> round(out2["estimate"], 9)
    4.0
    >>> wasserman_kernel_regression(0.0, [1.0], [1.0], 0.0)
    Traceback (most recent call last):
        ...
    ValueError: the bandwidth must be positive; got 0.0.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    xd = np.atleast_1d(np.asarray(x_data, dtype=float))
    yd = np.atleast_1d(np.asarray(y_data, dtype=float))
    h = float(h)
    if xd.size != yd.size:
        raise ValueError(f"x_data ({xd.size}) and y_data ({yd.size}) lengths differ.")
    if xd.size == 0:
        raise ValueError("kernel regression needs data.")
    if h <= 0:
        raise ValueError(f"the bandwidth must be positive; got {h}.")
    vals, eff = [], []
    for xi in x:
        w = np.exp(-0.5 * ((xi - xd) / h) ** 2)
        s = float(np.sum(w))
        eff.append(s)
        vals.append(float(np.sum(w * yd) / s) if s > 0 else float("nan"))
    return RichResult(payload={
        "estimate": vals[0], "values": vals, "effective_n": eff,
        "h": h, "n": int(xd.size),
        "method": "Nadaraya-Watson, Gaussian kernel; zero-weight -> nan"})


def cheatsheet():
    return "wsmcrk: m(x) = sum K((x-Xi)/h) Yi / sum K; Gaussian kernel"
