# morie.fn -- function file (rootcoder007/morie)
"""Elementary coverages are Beta(1, n)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["gibbons_elementary_coverage_beta"]


def gibbons_elementary_coverage_beta(n):
    r"""Corollary 2.11.1.1: each elementary coverage
    :math:`C_i = U_{(i)} - U_{(i-1)}` is distributed Beta(1, n), so

    .. math:: E(C_i) = \frac{1}{n + 1}, \qquad
              \mathrm{Var}(C_i) = \frac{n}{(n+1)^2(n+2)}.

    Every gap between adjacent order statistics of a uniform sample
    has the SAME marginal distribution regardless of position -- the
    exchangeability fact that makes block frequencies tractable.

    Parameters
    ----------
    n : int
        Sample size, at least 1.

    Returns
    -------
    RichResult
        keys: ``alpha`` (1), ``beta`` (n), ``mean``, ``var``, ``n``,
        ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Corollary 2.11.1.1.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}.")
    return RichResult(
        payload={
            "alpha": 1, "beta": n, "mean": 1.0 / (n + 1),
            "var": float(n / ((n + 1.0) ** 2 * (n + 2.0))), "n": n,
            "method": "C_i ~ Beta(1, n), position-free (Corollary 2.11.1.1)",
        }
    )


def cheatsheet():
    return "gb2111c: every elementary coverage ~ Beta(1, n); E = 1/(n+1)"
