# morie.fn -- function file (rootcoder007/morie)
"""Null variance of the Mann-Whitney U statistic."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["gibbons_mw_var"]


def gibbons_mw_var(m, n):
    r"""Null moments of Mann-Whitney U (Gibbons Ch. 6.6):

    .. math:: E(U) = \frac{mn}{2}, \qquad
              \mathrm{Var}(U) = \frac{mn(m + n + 1)}{12}.

    Parameters
    ----------
    m, n : int
        The two sample sizes, both >= 1.

    Returns
    -------
    RichResult
        keys: ``mean``, ``var``, ``sd``, ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 6.6.
    """
    m, n = int(m), int(n)
    if m < 1 or n < 1:
        raise ValueError("m and n must be at least 1.")
    var = m * n * (m + n + 1) / 12.0
    return RichResult(
        payload={"mean": m * n / 2.0, "var": float(var),
                 "sd": float(np.sqrt(var)), "m": m, "n": n,
                 "method": "E(U) = mn/2, Var(U) = mn(m+n+1)/12 (Ch. 6.6)"}
    )


def cheatsheet():
    return "gb661v: E = mn/2, Var = mn(m+n+1)/12"
