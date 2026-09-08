# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Variance Var(X) = E[(X-mu)^2]."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_variance"]


def wasserman_variance(x):
    """
    Variance Var(X) = E[(X-mu)^2].

    Formula: Var(X) = E[X^2] - E[X]^2, applied to the empirical
    distribution putting mass 1/n on each observation, so the payload
    ``estimate`` is the POPULATION variance (divisor n). The unbiased
    sample variance (divisor n-1) ships alongside as
    ``sample_variance``; neither is silently substituted for the
    other. The identity is computed via the stable centered form
    mean((x - mean)^2), which is algebraically equal.

    Parameters
    ----------
    x : array-like
        Input data (at least one observation).

    Returns
    -------
    result : dict
        Keys: estimate (population variance), sample_variance, mean,
        second_moment, sd, n, method.

    References
    ----------
    Wasserman (2004), Ch 3.

    Examples
    --------
    >>> out = wasserman_variance([1.0, 2.0, 3.0, 4.0])
    >>> out["estimate"]
    1.25
    >>> out["sample_variance"]
    1.6666666666666667
    >>> round(out["second_moment"] - out["mean"] ** 2, 12) == round(out["estimate"], 12)
    True
    >>> wasserman_variance([7.0])["estimate"]
    0.0
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = x.size
    if n == 0:
        raise ValueError("variance of an empty sample is undefined.")
    mu = float(np.mean(x))
    var_pop = float(np.mean((x - mu) ** 2))
    var_samp = float(np.var(x, ddof=1)) if n > 1 else 0.0
    return RichResult(payload={
        "estimate": var_pop, "sample_variance": var_samp, "mean": mu,
        "second_moment": float(np.mean(x ** 2)),
        "sd": float(np.sqrt(var_pop)), "n": int(n),
        "method": "Variance Var(X) = E[X^2] - E[X]^2"})


def cheatsheet():
    return "wsmvar: Var(X) = E[X^2] - E[X]^2 (population divisor n; n-1 alongside)"
