# morie.fn -- function file (rootcoder007/morie)
"""Kendall's W with tie correction."""

from . import _array_core as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["gibbons_concordance_w_ties"]


def gibbons_concordance_w_ties(rankings):
    r"""Tie-corrected coefficient of concordance.

    .. math:: W = \frac{S}{k^2 (n^3 - n)/12 - k \sum_i T_i}, \qquad
              T_i = \sum \frac{t(t^2 - 1)}{12}

    over judge i's tie groups (Gibbons Ch. 12.4). Judges who cannot
    distinguish objects get midranks; without the correction their
    ties deflate W below what the agreement warrants. Reduces to the
    plain W when no ranking has ties, which the tests assert.

    Parameters
    ----------
    rankings : array-like, shape (k, n)
        Rows are rankings, midranks allowed.

    Returns
    -------
    RichResult
        keys: ``W``, ``S``, ``tie_sum``, ``chi2``, ``df``,
        ``p_value``, ``k``, ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 12.4.
    """
    R = np.asarray(rankings, dtype=float)
    if R.ndim != 2:
        raise ValueError("rankings must be 2-D (k judges x n objects).")
    k, n = R.shape
    if k < 2 or n < 2:
        raise ValueError("need at least 2 judges and 2 objects.")
    col = R.sum(axis=0)
    S = float(np.sum((col - col.mean()) ** 2))
    T = 0.0
    for i in range(k):
        _, counts = np.unique(R[i], return_counts=True)
        T += float(np.sum(counts * (counts**2 - 1)) / 12.0)
    denom = k**2 * (n**3 - n) / 12.0 - k * T
    if denom <= 0:
        raise ValueError("every judge tied every object; W is undefined.")
    W = S / denom
    chi2 = k * (n - 1) * W
    return RichResult(
        payload={
            "W": float(W), "S": S, "tie_sum": T, "chi2": float(chi2),
            "df": int(n - 1), "p_value": float(stats.chi2.sf(chi2, n - 1)),
            "k": int(k), "n": int(n),
            "method": "Tie-corrected W = S/(k^2(n^3-n)/12 - k sum T_i)",
        }
    )


def cheatsheet():
    return "gb1241t: W with midrank tie sums; equals plain W when tie-free"
