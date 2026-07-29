# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Log-likelihood l(theta) = sum log f(X_i; theta)."""

import numpy as np

from ._richresult import RichResult
from .wsmlik import wasserman_likelihood

__all__ = ["wasserman_log_likelihood"]


def wasserman_log_likelihood(data, f, theta):
    """
    Log-likelihood of theta on i.i.d. data.

    Formula: l(theta) = sum_i log f(X_i; theta). Delegates to
    wsmlik's log-domain evaluation (single source of truth) and
    reports per-observation contributions so a single outlying
    observation's pull is visible.

    Parameters
    ----------
    data : array-like
        Sample (non-empty).
    f : callable or None
        Density f(x, theta); None = exponential(theta).
    theta : float
        Parameter value.

    Returns
    -------
    result : dict
        Keys: estimate (l), per_observation, likelihood, theta, n,
        method.

    References
    ----------
    Wasserman (2004), Ch 9, section 9.3.

    Examples
    --------
    >>> out = wasserman_log_likelihood([1.0, 2.0], None, 1.0)
    >>> round(out["estimate"], 12)
    -3.0
    >>> [round(v, 12) for v in out["per_observation"]]
    [-1.0, -2.0]
    >>> import math
    >>> abs(out["likelihood"] - math.e ** -3) < 1e-15
    True
    """
    core = wasserman_likelihood(data, f, theta)
    data_arr = np.atleast_1d(np.asarray(data, dtype=float))
    theta = float(theta)
    if f is None:
        f = lambda x, th: np.where(x >= 0, np.exp(-x / th) / th, 0.0)
    with np.errstate(divide="ignore"):
        per = [float(v) for v in np.log(np.asarray(f(data_arr, theta), dtype=float))]
    return RichResult(payload={
        "estimate": core["log_likelihood"], "per_observation": per,
        "likelihood": core["estimate"], "theta": theta,
        "n": int(data_arr.size),
        "method": "l(theta) = sum log f(X_i;theta)"})


def cheatsheet():
    return "wsmllk: sum log f, per-observation terms in payload; delegates to wsmlik"
