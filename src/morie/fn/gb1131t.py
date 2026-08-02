# morie.fn -- function file (rootcoder007/morie)
"""Spearman's coefficient with tie correction."""

from . import _array_core as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["gibbons_spearman_ties"]


def gibbons_spearman_ties(x, y):
    r"""Tie-corrected Spearman rank correlation.

    .. math:: r_s = \frac{(n^3 - n)/6 - \sum d_i^2 - T_x - T_y}
              {\sqrt{[(n^3-n)/6 - 2T_x]\,[(n^3-n)/6 - 2T_y]}}

    with tie sums :math:`T = \sum t(t^2 - 1)/12` over each variable's
    tie groups (Gibbons Ch. 11.3). With midranks this equals the
    Pearson correlation of the ranks, which is the identity the test
    pins against scipy.

    Parameters
    ----------
    x, y : array-like, shape (n,)
        Paired observations, n >= 3.

    Returns
    -------
    RichResult
        keys: ``r_s``, ``sum_d2``, ``T_x``, ``T_y``, ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 11.3.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    if y.size != n:
        raise ValueError("x and y must have the same length.")
    if n < 3:
        raise ValueError(f"need at least 3 pairs, got {n}.")

    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    d2 = float(np.sum((rx - ry) ** 2))

    def tie_sum(v):
        _, counts = np.unique(v, return_counts=True)
        return float(np.sum(counts * (counts**2 - 1)) / 12.0)

    Tx, Ty = tie_sum(x), tie_sum(y)
    base = (n**3 - n) / 6.0
    denom = np.sqrt((base - 2 * Tx) * (base - 2 * Ty))
    if denom == 0:
        raise ValueError("a variable is entirely tied; r_s is undefined.")
    return RichResult(
        payload={
            "r_s": float((base - d2 - Tx - Ty) / denom),
            "sum_d2": d2, "T_x": Tx, "T_y": Ty, "n": int(n),
            "method": "Tie-corrected Spearman r_s (Gibbons Ch. 11.3)",
        }
    )


def cheatsheet():
    return "gb1131t: tie-corrected r_s == Pearson correlation of midranks"
