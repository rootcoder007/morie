# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DKW confidence band for the eCDF."""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_dkw_cb"]


def wasserman_dkw_cb(data, alpha):
    """
    Dvoretzky-Kiefer-Wolfowitz 1-alpha confidence band for F.

    Formula: L(x) = max(F_n(x) - eps, 0), U(x) = min(F_n(x) + eps, 1)
    with eps = sqrt(log(2/alpha) / (2n)). Evaluated at the sorted
    sample points, where F_n jumps.

    Parameters
    ----------
    data : array-like
        Sample (non-empty).
    alpha : float
        Level in (0, 1).

    Returns
    -------
    result : dict
        Keys: estimate (eps), lower, upper, ecdf, x_sorted, alpha,
        n, method.

    References
    ----------
    Wasserman (2004), Ch 7, Theorem 7.5.

    Examples
    --------
    n=2, alpha=2/e makes eps = 0.5 exactly: log(2/alpha) = 1, so
    eps = sqrt(1/4).

    >>> import math
    >>> out = wasserman_dkw_cb([1.0, 2.0], 2.0 / math.e)
    >>> round(out["estimate"], 12)
    0.5
    >>> out["ecdf"]
    [0.5, 1.0]
    >>> out["lower"]
    [0.0, 0.5]
    >>> out["upper"]
    [1.0, 1.0]
    >>> wasserman_dkw_cb([1.0], 1.5)
    Traceback (most recent call last):
        ...
    ValueError: alpha must lie in (0, 1); got 1.5.
    """
    data = np.sort(np.atleast_1d(np.asarray(data, dtype=float)))
    alpha = float(alpha)
    n = data.size
    if n == 0:
        raise ValueError("the DKW band of an empty sample is undefined.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha}.")
    eps = math.sqrt(math.log(2.0 / alpha) / (2.0 * n))
    ecdf = np.arange(1, n + 1) / float(n)
    lower = np.maximum(ecdf - eps, 0.0)
    upper = np.minimum(ecdf + eps, 1.0)
    return RichResult(payload={
        "estimate": float(eps),
        "lower": [float(v) for v in lower],
        "upper": [float(v) for v in upper],
        "ecdf": [float(v) for v in ecdf],
        "x_sorted": [float(v) for v in data],
        "alpha": alpha, "n": int(n),
        "method": "DKW band F_n +/- sqrt(log(2/alpha)/(2n))"})


def cheatsheet():
    return "wsmcb: eps = sqrt(log(2/alpha)/(2n)); band clipped to [0,1]"


# compact alias per ledger/NAMING.md
wassermandkwcb = wasserman_dkw_cb
