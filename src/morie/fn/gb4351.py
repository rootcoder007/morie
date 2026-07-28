# morie.fn -- function file (rootcoder007/morie)
"""Chi-square form of the one-sided K-S limit."""

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["gibbons_ks_chi2_approx"]


def gibbons_ks_chi2_approx(n, Dplus):
    r"""Corollary 4.3.5.1: since P(D_n^+ > d/sqrt n) -> exp(-2d^2)
    and the chi-square(2) survival function is exp(-x/2),

    .. math:: 4 n (D_n^+)^2 \;\to_d\; \chi^2_2.

    The same limit as Theorem 4.3.5 wearing chi-square clothes -- the
    p-values are algebraically identical, which the tests assert.

    Parameters
    ----------
    n : int
        Sample size.
    Dplus : float > 0
        Observed one-sided statistic.

    Returns
    -------
    RichResult
        keys: ``chi2_stat`` (4 n D+^2), ``df`` (2), ``p_value``,
        ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Corollary 4.3.5.1.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}.")
    Dplus = float(Dplus)
    if Dplus <= 0:
        raise ValueError(f"Dplus must be positive, got {Dplus}.")
    stat = 4.0 * n * Dplus**2
    return RichResult(
        payload={
            "chi2_stat": float(stat), "df": 2,
            "p_value": float(stats.chi2.sf(stat, 2)), "n": n,
            "method": "4n(D+)^2 -> chi2(2) (Gibbons Corollary 4.3.5.1)",
        }
    )


def cheatsheet():
    return "gb4351: 4n(D+)^2 ~ chi2(2); identical p to exp(-2nD+^2)"
