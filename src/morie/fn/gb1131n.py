# morie.fn -- function file (rootcoder007/morie)
"""Spearman asymptotic null distribution."""

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["gibbons_spearman_asymp"]


def gibbons_spearman_asymp(r_s, n):
    r"""Large-sample null test for Spearman's coefficient.

    Under independence :math:`E(R) = 0` and
    :math:`\mathrm{Var}(R) = 1/(n-1)`, so

    .. math:: Z = r_s \sqrt{n - 1} \;\to_d\; N(0, 1)

    (Gibbons Ch. 11.3). The approximation is stated as usable for
    n > 10; below that the exact permutation distribution
    (:mod:`morie.fn.gb_sp2`) is the honest tool, and this function
    says so in a returned flag rather than silently applying the
    normal anyway.

    Parameters
    ----------
    r_s : float in [-1, 1]
        Observed Spearman coefficient.
    n : int
        Sample size, at least 3.

    Returns
    -------
    RichResult
        keys: ``z``, ``p_two_sided``, ``p_one_sided``, ``var_null``,
        ``large_sample_ok`` (n > 10), ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 11.3.
    """
    r_s = float(r_s)
    n = int(n)
    if not -1 <= r_s <= 1:
        raise ValueError(f"r_s must lie in [-1, 1], got {r_s}.")
    if n < 3:
        raise ValueError(f"n must be at least 3, got {n}.")
    z = r_s * np.sqrt(n - 1.0)
    return RichResult(
        payload={
            "z": float(z),
            "p_two_sided": float(2 * stats.norm.sf(abs(z))),
            "p_one_sided": float(stats.norm.sf(z)),
            "var_null": 1.0 / (n - 1), "large_sample_ok": bool(n > 10),
            "n": n, "method": "Z = r_s sqrt(n-1) ~ N(0,1) (Gibbons Ch. 11.3)",
        }
    )


def cheatsheet():
    return "gb1131n: Z = r_s sqrt(n-1); flag says when n is too small for it"
