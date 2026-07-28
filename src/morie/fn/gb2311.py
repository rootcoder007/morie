# morie.fn -- function file (rootcoder007/morie)
"""Moments of the empirical distribution function."""

import numpy as np

from ._richresult import RichResult

__all__ = ["gibbons_edf_mean_var"]


def gibbons_edf_mean_var(F_x, n):
    r"""Corollary 2.3.1.1: at any fixed x, :math:`n S_n(x)` is
    Binomial(n, F(x)), so

    .. math:: E[S_n(x)] = F(x), \qquad
              \mathrm{Var}[S_n(x)] = \frac{F(x)(1 - F(x))}{n}.

    The EDF is unbiased at every point with variance shrinking at
    1/n -- pointwise; the uniform statement is Glivenko-Cantelli.

    Parameters
    ----------
    F_x : float or array-like in [0, 1]
        The true CDF value(s) F(x).
    n : int
        Sample size.

    Returns
    -------
    RichResult
        keys: ``mean``, ``var``, ``binomial_n``, ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Corollary 2.3.1.1.
    """
    F = np.asarray(F_x, dtype=float)
    if np.any((F < 0) | (F > 1)):
        raise ValueError("F(x) values must lie in [0, 1].")
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}.")
    return RichResult(
        payload={
            "mean": F, "var": F * (1 - F) / n, "binomial_n": n, "n": n,
            "method": "n S_n(x) ~ Bin(n, F(x)) (Gibbons Corollary 2.3.1.1)",
        }
    )


def cheatsheet():
    return "gb2311: E = F, Var = F(1-F)/n; pointwise, GC does uniform"
