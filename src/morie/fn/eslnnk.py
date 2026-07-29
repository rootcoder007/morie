# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Nadaraya-Watson kernel smoother (ESL Ch 6.1)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["esl_nadaraya_watson"]

_KERNELS = ("epanechnikov", "tri-cube", "gaussian")


def _kernel_weights(t, kernel):
    """t is |x0 - xi| / lambda."""
    if kernel == "gaussian":
        return np.exp(-0.5 * t ** 2)
    inside = t <= 1.0
    if kernel == "epanechnikov":
        return np.where(inside, 0.75 * (1.0 - t ** 2), 0.0)
    return np.where(inside, (1.0 - t ** 3) ** 3, 0.0)      # tri-cube


def esl_nadaraya_watson(x0, x_data, y_data, lambda_, kernel="epanechnikov"):
    """
    Nadaraya-Watson: f(x0) = sum K_l(x0,xi) yi / sum K_l(x0,xi).

    ESL Ch 6.1 uses the Epanechnikov kernel as its default, so that is
    the default here; the Gaussian is also offered and differs in one
    way that matters: it has INFINITE support, so every observation
    always contributes and the estimate is never undefined. The
    compact kernels (Epanechnikov, tri-cube) are zero beyond lambda,
    which means a query point with no neighbours inside the window has
    no estimate at all — returned as nan rather than invented, with
    the effective sample size reported so the emptiness is visible.

    The Gaussian's infinite support is a mathematical property, not a
    numerical one: at |x0 - xi| / lambda beyond roughly 38 the weight
    exp(-t^2/2) underflows to exactly zero in float64, so a far enough
    query returns nan there too. That limit is reported through
    effective_n rather than being asserted away.

    The known weakness, stated because it is the reason local linear
    regression exists: this estimator is biased at the BOUNDARY, where
    the window becomes one-sided and the local mean is pulled toward
    the interior.

    Parameters
    ----------
    x0 : array-like
        Query point(s).
    x_data, y_data : array-like
        Training pairs of equal length.
    lambda_ : float
        Bandwidth, > 0.
    kernel : str
        "epanechnikov" (default), "tri-cube", or "gaussian".

    Returns
    -------
    result : dict
        Keys: estimate (fit at the first x0), values, effective_n,
        n_in_window, lambda, kernel, n, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 6.1 (Eq. 6.2, 6.4).

    Examples
    --------
    A constant response is reproduced exactly, whatever the kernel:

    >>> xs = [0.0, 1.0, 2.0, 3.0]
    >>> esl_nadaraya_watson(1.5, xs, [5.0, 5.0, 5.0, 5.0], 1.0)["estimate"]
    5.0
    >>> out = esl_nadaraya_watson(1.5, xs, [0.0, 1.0, 2.0, 3.0], 1.0)
    >>> round(out["estimate"], 12)
    1.5
    >>> out["n_in_window"]
    [2]

    A point far outside every compact window has no estimate:

    >>> esl_nadaraya_watson(6.0, xs, [0.0, 1.0, 2.0, 3.0], 1.0)["estimate"]
    nan

    The Gaussian still reaches that point, because its support is
    unbounded:

    >>> import math
    >>> g = esl_nadaraya_watson(6.0, xs, [0.0, 1.0, 2.0, 3.0], 1.0, "gaussian")
    >>> math.isnan(g["estimate"])
    False
    >>> round(g["estimate"], 6)      # pulled below 3 by the far tail
    2.970042

    But far enough away even the Gaussian underflows to nothing:

    >>> far = esl_nadaraya_watson(100.0, xs, [0.0, 1.0, 2.0, 3.0], 1.0, "gaussian")
    >>> far["effective_n"]
    [0.0]
    >>> math.isnan(far["estimate"])
    True
    """
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    xd = np.atleast_1d(np.asarray(x_data, dtype=float))
    yd = np.atleast_1d(np.asarray(y_data, dtype=float))
    lam = float(lambda_)
    if xd.size != yd.size:
        raise ValueError(f"x_data ({xd.size}) and y_data ({yd.size}) lengths differ.")
    if xd.size == 0:
        raise ValueError("the smoother needs data.")
    if lam <= 0:
        raise ValueError(f"the bandwidth must be positive; got {lam}.")
    if kernel not in _KERNELS:
        raise ValueError(f"kernel must be one of {_KERNELS}; got '{kernel}'.")
    vals, eff, cnt = [], [], []
    for q in x0:
        w = _kernel_weights(np.abs(q - xd) / lam, kernel)
        s = float(np.sum(w))
        eff.append(s)
        cnt.append(int(np.sum(w > 0)))
        vals.append(float(np.sum(w * yd) / s) if s > 0 else float("nan"))
    return RichResult(payload={
        "estimate": vals[0], "values": vals, "effective_n": eff,
        "n_in_window": cnt, "lambda": lam, "kernel": kernel, "n": int(xd.size),
        "method": f"Nadaraya-Watson, {kernel} kernel; empty window -> nan; biased at boundaries"})


def cheatsheet():
    return "eslnnk: Epanechnikov default (compact); empty window -> nan; boundary-biased"
