# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Central limit theorem standardisation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_clt"]


def wasserman_clt(data):
    """
    CLT standardised mean of the sample.

    Formula: Z_n = sqrt(n) (X_bar - mu) / sigma ~> N(0, 1). Without
    the true mu/sigma, the classical plug-in studentisation is
    reported: mu is estimated by X_bar itself only in the SE (the
    payload carries the ingredients so any reference value can be
    standardised): z = sqrt(n) (X_bar - mu0) / s for a caller-chosen
    mu0 defaults to mu0 = 0.

    Parameters
    ----------
    data : array-like
        Sample with n >= 2 (a sample sd needs 2 points).

    Returns
    -------
    result : dict
        Keys: estimate (z for mu0 = 0), mean, sd (sample, ddof=1),
        se, n, method.

    References
    ----------
    Wasserman (2004), Ch 5, Theorem 5.8.

    Examples
    --------
    >>> out = wasserman_clt([1.0, 2.0, 3.0, 4.0])
    >>> out["mean"]
    2.5
    >>> round(out["sd"], 15)
    1.290994448735806
    >>> round(out["estimate"], 12) == round(2.5 / (out["sd"] / 2.0), 12)
    True
    >>> wasserman_clt([1.0])
    Traceback (most recent call last):
        ...
    ValueError: CLT standardisation needs n >= 2 for a sample sd.
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = data.size
    if n < 2:
        raise ValueError("CLT standardisation needs n >= 2 for a sample sd.")
    xbar = float(np.mean(data))
    s = float(np.std(data, ddof=1))
    if s == 0:
        raise ValueError("a constant sample has sd 0; z is undefined.")
    se = s / float(np.sqrt(n))
    return RichResult(payload={
        "estimate": float(xbar / se), "mean": xbar, "sd": s, "se": se,
        "n": int(n),
        "method": "CLT z = sqrt(n)(X_bar - mu0)/s, mu0 = 0"})


def cheatsheet():
    return "wsmclt: z = sqrt(n)(X_bar - mu0)/s with mu0 = 0; ingredients in payload"


# compact alias per ledger/NAMING.md
wassermanclt = wasserman_clt
