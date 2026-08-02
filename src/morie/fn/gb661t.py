# morie.fn -- function file (rootcoder007/morie)
"""Mann-Whitney with the tie-corrected variance."""

from . import _array_core as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["gibbons_mw_ties"]


def gibbons_mw_ties(x, y):
    r"""Mann-Whitney U test with midranks and the tie-corrected null
    variance (Gibbons Ch. 6.6):

    .. math:: \mathrm{Var}(U) = \frac{mn}{12}\Big[(m + n + 1)
              - \frac{\sum t(t^2 - 1)}{(m + n)(m + n - 1)}\Big],

    the sum running over the tie multiplicities of the COMBINED
    sample. Ties shrink the variance; using the tie-free formula on
    tied data makes the test conservative, not liberal -- worth
    knowing when deciding whether an uncorrected p-value can be
    trusted.

    Parameters
    ----------
    x, y : array-like
        The two samples.

    Returns
    -------
    RichResult
        keys: ``U``, ``mean``, ``var_corrected``, ``var_uncorrected``,
        ``z``, ``p_two_sided``, ``tie_sum``, ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 6.6.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m, n = x.size, y.size
    if m < 1 or n < 1:
        raise ValueError("both samples must be non-empty.")
    N = m + n
    combined = np.r_[x, y]
    ranks = stats.rankdata(combined)
    W = float(ranks[:m].sum())
    U = W - m * (m + 1) / 2.0
    _, counts = np.unique(combined, return_counts=True)
    tie_sum = float(np.sum(counts * (counts**2 - 1)))
    var0 = m * n * (N + 1) / 12.0
    var = m * n / 12.0 * ((N + 1) - tie_sum / (N * (N - 1)))
    if var <= 0:
        raise ValueError("all observations tied; the test is degenerate.")
    mean = m * n / 2.0
    z = (U - mean) / np.sqrt(var)
    return RichResult(
        payload={
            "U": float(U), "mean": mean, "var_corrected": float(var),
            "var_uncorrected": float(var0), "z": float(z),
            "p_two_sided": float(2 * stats.norm.sf(abs(z))),
            "tie_sum": tie_sum, "m": m, "n": n,
            "method": "Mann-Whitney with tie-corrected variance (Gibbons Ch. 6.6)",
        }
    )


def cheatsheet():
    return "gb661t: ties shrink Var(U); uncorrected formula is conservative"
