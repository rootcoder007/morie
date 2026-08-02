# morie.fn -- function file (rootcoder007/morie)
"""Kendall's coefficient of concordance W."""

from . import _array_core as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["gibbons_concordance_w"]


def gibbons_concordance_w(rankings):
    r"""Kendall's W for k complete rankings of n objects.

    .. math:: W = \frac{12 S}{k^2 (n^3 - n)}, \qquad
              S = \sum_j \big(R_j - \bar R\big)^2

    (Gibbons eq. 12.4.4, PDF-verified). W = 1 means the k judges
    agree perfectly; W = 0 is what independent random rankings
    approach. The chi-square test uses
    :math:`k(n-1)W \sim \chi^2_{n-1}` for k not too small.

    Parameters
    ----------
    rankings : array-like, shape (k, n)
        Each row is one judge's ranking of the n objects (1..n).

    Returns
    -------
    RichResult
        keys: ``W``, ``S``, ``chi2``, ``df``, ``p_value``,
        ``mean_spearman`` ((kW - 1)/(k - 1), the average pairwise
        rho), ``k``, ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 12.4,
    eq. (12.4.4) and (12.4.7).
    """
    R = np.asarray(rankings, dtype=float)
    if R.ndim != 2:
        raise ValueError("rankings must be 2-D (k judges x n objects).")
    k, n = R.shape
    if k < 2 or n < 2:
        raise ValueError("need at least 2 judges and 2 objects.")
    col = R.sum(axis=0)
    S = float(np.sum((col - col.mean()) ** 2))
    W = 12.0 * S / (k**2 * (n**3 - n))
    chi2 = k * (n - 1) * W
    return RichResult(
        payload={
            "W": float(W), "S": S, "chi2": float(chi2), "df": int(n - 1),
            "p_value": float(stats.chi2.sf(chi2, n - 1)),
            "mean_spearman": float((k * W - 1) / (k - 1)),
            "k": int(k), "n": int(n),
            "method": "Kendall W = 12S/(k^2(n^3-n)) (Gibbons eq. 12.4.4)",
        }
    )


def cheatsheet():
    return "gb1241: W = 12S/(k^2(n^3-n)); mean rho = (kW-1)/(k-1)"
