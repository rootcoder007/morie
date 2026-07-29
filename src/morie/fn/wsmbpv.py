# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bootstrap pivotal confidence interval."""

import numpy as np

from ._richresult import RichResult
from .wsmbpc import _boot_replicates, _type1_quantile

__all__ = ["wasserman_bootstrap_pivotal"]


def wasserman_bootstrap_pivotal(data, T, B, alpha, seed=13):
    """
    Bootstrap pivotal interval.

    Formula: (2 theta_hat - q*_{1-alpha/2}, 2 theta_hat - q*_{alpha/2})
    where q* are type-1 quantiles of the bootstrap replicates. Unlike
    the percentile interval, this one inverts the pivot
    theta_hat - theta, so the UPPER replicate quantile sets the LOWER
    endpoint. Same LCG resampling as wsmbpc, so on identical inputs
    the two intervals come from the identical replicate set.

    Parameters
    ----------
    data : array-like
        Sample (non-empty).
    T : callable or None
        Statistic (None = sample mean).
    B : int
        Replications, >= 2.
    alpha : float
        Level in (0, 1).
    seed : int
        LCG seed (default 13).

    Returns
    -------
    result : dict
        Keys: estimate (theta_hat), lower, upper, alpha, B, n,
        method.

    References
    ----------
    Wasserman (2004), Ch 8, section 8.3.

    Examples
    --------
    Pivotal endpoints are the percentile endpoints reflected about
    theta_hat:

    >>> from morie.fn.wsmbpc import wasserman_bootstrap_percentile
    >>> d = [1.0, 2.0, 3.0, 4.0]
    >>> pv = wasserman_bootstrap_pivotal(d, None, 400, 0.10)
    >>> pc = wasserman_bootstrap_percentile(d, None, 400, 0.10)
    >>> round(pv["lower"], 12) == round(2 * 2.5 - pc["upper"], 12)
    True
    >>> round(pv["upper"], 12) == round(2 * 2.5 - pc["lower"], 12)
    True
    >>> pv["lower"] < pv["upper"]
    True
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    alpha = float(alpha)
    B = int(B)
    if data.size == 0:
        raise ValueError("the bootstrap of an empty sample is undefined.")
    if B < 2:
        raise ValueError(f"the bootstrap needs B >= 2; got {B}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha}.")
    if T is None:
        T = lambda a: float(np.mean(a))
    theta = float(T(data))
    reps = np.sort(_boot_replicates(data, T, B, seed))
    q_lo = _type1_quantile(reps, alpha / 2.0)
    q_hi = _type1_quantile(reps, 1.0 - alpha / 2.0)
    return RichResult(payload={
        "estimate": theta,
        "lower": float(2.0 * theta - q_hi),
        "upper": float(2.0 * theta - q_lo),
        "alpha": alpha, "B": B, "n": int(data.size),
        "method": "bootstrap pivotal CI (2 theta - q*_{1-a/2}, 2 theta - q*_{a/2})"})


def cheatsheet():
    return "wsmbpv: reflect replicate quantiles about theta_hat; upper q* sets lower end"
