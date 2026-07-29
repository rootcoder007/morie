# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bootstrap percentile confidence interval."""

import numpy as np

from ._richresult import RichResult
from .wsmnpb import _lcg_uniforms

__all__ = ["wasserman_bootstrap_percentile"]


def _boot_replicates(data, T, B, seed):
    n = data.size
    u = _lcg_uniforms(B * n, seed)
    idx = np.minimum((u * n).astype(int), n - 1).reshape(B, n)
    return np.array([float(T(data[row])) for row in idx])


def _type1_quantile(sorted_vals, p):
    n = sorted_vals.size
    return float(sorted_vals[int(np.ceil(p * n)) - 1])


def wasserman_bootstrap_percentile(data, T, B, alpha, seed=13):
    """
    Bootstrap percentile interval (q*_{alpha/2}, q*_{1-alpha/2}).

    Formula: the interval endpoints are the alpha/2 and 1 - alpha/2
    type-1 quantiles of the bootstrap replicates theta*_1..theta*_B.
    Resampling uses the shared exact-integer LCG for cross-language
    determinism; quantiles are order statistics (no interpolation),
    matching wsmqtl's inf definition.

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
    >>> out = wasserman_bootstrap_percentile([1.0, 2.0, 3.0, 4.0], None, 400, 0.10)
    >>> out["estimate"]
    2.5
    >>> out["lower"] < 2.5 < out["upper"]
    True
    >>> out["lower"] >= 1.0 and out["upper"] <= 4.0
    True
    >>> wasserman_bootstrap_percentile([1.0, 2.0], None, 100, 0.0)
    Traceback (most recent call last):
        ...
    ValueError: alpha must lie in (0, 1); got 0.0.
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
    reps = np.sort(_boot_replicates(data, T, B, seed))
    lo = _type1_quantile(reps, alpha / 2.0)
    hi = _type1_quantile(reps, 1.0 - alpha / 2.0)
    return RichResult(payload={
        "estimate": float(T(data)), "lower": lo, "upper": hi,
        "alpha": alpha, "B": B, "n": int(data.size),
        "method": "bootstrap percentile CI, type-1 quantiles, LCG"})


def cheatsheet():
    return "wsmbpc: (q*_{a/2}, q*_{1-a/2}) of LCG bootstrap replicates"
