# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Nonparametric bootstrap standard error."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_nonparametric_boot"]


def _lcg_uniforms(count, seed=13):
    """Shared exact-integer LCG (cross-language deterministic)."""
    s = int(seed)
    out = np.empty(count)
    for i in range(count):
        s = (1664525 * s + 1013904223) % 2 ** 32
        out[i] = (s + 0.5) / 2 ** 32
    return out


def wasserman_nonparametric_boot(data, T, B, seed=13):
    """
    Nonparametric bootstrap SE of a statistic T.

    Formula: se_boot = sqrt((1/B) sum_b (theta*_b - theta*_bar)^2).
    Resampling indices come from the shared exact-integer LCG
    (s -> (1664525 s + 1013904223) mod 2^32), so runs are
    reproducible bit-for-bit across languages. Wasserman's (1/B)
    divisor is used, NOT 1/(B-1); both are reported.

    Parameters
    ----------
    data : array-like
        Sample (non-empty).
    T : callable
        Statistic mapping a 1-D array to a float. None means the
        sample mean.
    B : int
        Bootstrap replications, >= 2.
    seed : int
        LCG seed (default 13).

    Returns
    -------
    result : dict
        Keys: estimate (theta_hat on the data), se (1/B divisor),
        se_unbiased (1/(B-1)), replicates_mean, B, n, method.

    References
    ----------
    Wasserman (2004), Ch 8, section 8.2.

    Examples
    --------
    >>> out = wasserman_nonparametric_boot([1.0, 2.0, 3.0, 4.0], None, 200)
    >>> out["estimate"]
    2.5
    >>> 0.3 < out["se"] < 0.85
    True
    >>> out["B"]
    200
    >>> wasserman_nonparametric_boot([1.0], None, 1)
    Traceback (most recent call last):
        ...
    ValueError: the bootstrap needs B >= 2; got 1.
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = data.size
    B = int(B)
    if n == 0:
        raise ValueError("the bootstrap of an empty sample is undefined.")
    if B < 2:
        raise ValueError(f"the bootstrap needs B >= 2; got {B}.")
    if T is None:
        T = lambda a: float(np.mean(a))
    u = _lcg_uniforms(B * n, seed)
    idx = np.minimum((u * n).astype(int), n - 1).reshape(B, n)
    reps = np.array([float(T(data[row])) for row in idx])
    rbar = float(np.mean(reps))
    se_b = float(np.sqrt(np.mean((reps - rbar) ** 2)))
    se_u = float(np.sqrt(np.sum((reps - rbar) ** 2) / (B - 1)))
    return RichResult(payload={
        "estimate": float(T(data)), "se": se_b, "se_unbiased": se_u,
        "replicates_mean": rbar, "B": B, "n": int(n),
        "method": "nonparametric bootstrap, LCG resampling, 1/B divisor"})


def cheatsheet():
    return "wsmnpb: se_boot = sqrt(mean (theta*_b - bar)^2), LCG-deterministic"
