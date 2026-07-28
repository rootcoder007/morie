# morie.fn -- function file (rootcoder007/morie)
"""Empirical quantile function."""

import numpy as np

from ._richresult import RichResult

__all__ = ["gibbons_emp_quantile"]


def gibbons_emp_quantile(u, data):
    r"""Section 2.3.1: the empirical quantile function is the
    left-continuous inverse of the EDF,

    .. math:: Q_n(u) = X_{(i)} \quad\text{for}\quad
              \frac{i-1}{n} < u \le \frac{i}{n},

    i.e. the smallest order statistic whose EDF value reaches u --
    NOT an interpolated quantile: Q_n is a step function taking only
    observed values, which is what makes it distribution-free.

    Parameters
    ----------
    u : float or array-like in (0, 1]
        Probability levels.
    data : array-like
        Sample.

    Returns
    -------
    RichResult
        keys: ``quantile``, ``index`` (1-based i), ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 2.3.1.
    """
    x = np.sort(np.asarray(data, dtype=float).ravel())
    n = x.size
    if n < 1:
        raise ValueError("data must be non-empty.")
    uu = np.atleast_1d(np.asarray(u, dtype=float))
    if np.any((uu <= 0) | (uu > 1)):
        raise ValueError("u must lie in (0, 1].")
    idx = np.ceil(uu * n).astype(int)  # smallest i with i/n >= u
    q = x[idx - 1]
    scalar = np.isscalar(u) or np.ndim(u) == 0
    return RichResult(
        payload={
            "quantile": float(q[0]) if scalar else q,
            "index": int(idx[0]) if scalar else idx,
            "n": int(n),
            "method": "Q_n(u) = X_(ceil(nu)), step function (Gibbons Ch. 2.3.1)",
        }
    )


def cheatsheet():
    return "gb_eqf: Q_n(u) = X_(ceil(nu)); a step function, never interpolated"
