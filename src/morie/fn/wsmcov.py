# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Covariance Cov(X,Y) = E[XY] - E[X]E[Y]."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_covariance"]


def wasserman_covariance(x, y):
    """
    Covariance Cov(X,Y) = E[XY] - E[X]E[Y].

    Formula applied to the empirical distribution (divisor n), with
    the unbiased n-1 version alongside as ``sample_covariance`` and
    the correlation for scale-free reading. Computed via the centered
    form mean((x-xbar)(y-ybar)), algebraically equal to
    E[XY] - E[X]E[Y] but numerically stabler.

    Parameters
    ----------
    x, y : array-like
        Paired samples of equal length (>= 1).

    Returns
    -------
    result : dict
        Keys: estimate (population covariance), sample_covariance,
        correlation (nan for constant input), mean_x, mean_y, n,
        method.

    References
    ----------
    Wasserman (2004), Ch 4.

    Examples
    --------
    >>> out = wasserman_covariance([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    >>> round(out["estimate"], 12)
    1.333333333333
    >>> out["sample_covariance"]
    2.0
    >>> out["correlation"]
    1.0
    >>> wasserman_covariance([1.0, 2.0], [5.0, 5.0])["correlation"]
    nan
    >>> wasserman_covariance([1.0], [2.0, 3.0])
    Traceback (most recent call last):
        ...
    ValueError: paired samples must have equal length; got 1 and 2.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    if x.size != y.size:
        raise ValueError(f"paired samples must have equal length; got {x.size} and {y.size}.")
    n = x.size
    if n == 0:
        raise ValueError("covariance of an empty sample is undefined.")
    mx, my = float(np.mean(x)), float(np.mean(y))
    cov_pop = float(np.mean((x - mx) * (y - my)))
    cov_samp = cov_pop * n / (n - 1) if n > 1 else 0.0
    sx = float(np.sqrt(np.mean((x - mx) ** 2)))
    sy = float(np.sqrt(np.mean((y - my) ** 2)))
    corr = cov_pop / (sx * sy) if sx > 0 and sy > 0 else float("nan")
    return RichResult(payload={
        "estimate": cov_pop, "sample_covariance": cov_samp,
        "correlation": corr, "mean_x": mx, "mean_y": my, "n": int(n),
        "method": "Cov(X,Y) = E[XY] - E[X]E[Y] (population divisor n)"})


def cheatsheet():
    return "wsmcov: Cov = mean((x-xbar)(y-ybar)); n-1 version + correlation alongside"
