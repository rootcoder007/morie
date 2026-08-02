# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""KDE reference bandwidth."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_kde_bandwidth"]


def wasserman_kde_bandwidth(data):
    """
    Normal-reference KDE bandwidth.

    Formula: h_opt = c sigma n^{-1/5}. Two standard constants ship:
    Silverman's rule-of-thumb h = 0.9 min(s, IQR/1.34) n^{-1/5}
    (the ``estimate``) and the normal-reference
    h = 1.05922... s n^{-1/5} (Wasserman Ch 20's (4/(3n))^{1/5} s).
    IQR uses type-1 quantiles for determinism.

    Parameters
    ----------
    data : array-like
        Sample, n >= 2, not constant.

    Returns
    -------
    result : dict
        Keys: estimate (Silverman), h_normal_reference, sd, iqr, n,
        method.

    References
    ----------
    Wasserman (2004), Ch 20, section 20.3; Silverman (1986).

    Examples
    --------
    >>> d = list(range(1, 101))
    >>> out = wasserman_kde_bandwidth(d)
    >>> import numpy as np
    >>> s = float(np.std(d, ddof=1))
    >>> round(out["h_normal_reference"], 12) == round((4.0 / (3 * 100)) ** 0.2 * s, 12)
    True
    >>> out["estimate"] < out["h_normal_reference"]
    True
    >>> wasserman_kde_bandwidth([5.0, 5.0])
    Traceback (most recent call last):
        ...
    ValueError: a constant sample has no meaningful bandwidth.
    """
    data = np.sort(np.atleast_1d(np.asarray(data, dtype=float)))
    n = data.size
    if n < 2:
        raise ValueError("a bandwidth needs n >= 2.")
    s = float(np.std(data, ddof=1))
    if s == 0:
        raise ValueError("a constant sample has no meaningful bandwidth.")
    q1 = data[int(np.ceil(0.25 * n)) - 1]
    q3 = data[int(np.ceil(0.75 * n)) - 1]
    iqr = float(q3 - q1)
    spread = min(s, iqr / 1.34) if iqr > 0 else s
    return RichResult(payload={
        "estimate": float(0.9 * spread * n ** -0.2),
        "h_normal_reference": float((4.0 / (3.0 * n)) ** 0.2 * s),
        "sd": s, "iqr": iqr, "n": int(n),
        "method": "Silverman 0.9 min(s, IQR/1.34) n^-1/5; normal reference alongside"})


def cheatsheet():
    return "wsmkbw: Silverman + (4/(3n))^{1/5} s; type-1 IQR for determinism"
