# morie.fn -- function file (rootcoder007/morie)
"""Wilcoxon rank-sum with the tie-corrected variance."""

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["gibbons_wrs_ties"]


def gibbons_wrs_ties(x, y):
    r"""Wilcoxon rank-sum test with midranks and the tie-corrected
    null variance (Gibbons Ch. 8.2):

    .. math:: \mathrm{Var}(W) = \frac{mn}{12}\Big[(N + 1)
              - \frac{\sum t(t^2 - 1)}{N(N - 1)}\Big], \qquad
              N = m + n.

    W and U differ by the constant m(m+1)/2, so their tie-corrected
    variances are IDENTICAL -- returned side by side with the U-form
    module so the equivalence is checkable, not folklore.

    Parameters
    ----------
    x, y : array-like
        The two samples; W sums the ranks of x.

    Returns
    -------
    RichResult
        keys: ``W``, ``mean`` (m(N+1)/2), ``var_corrected``, ``z``,
        ``p_two_sided``, ``tie_sum``, ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 8.2.
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
    _, counts = np.unique(combined, return_counts=True)
    tie_sum = float(np.sum(counts * (counts**2 - 1)))
    var = m * n / 12.0 * ((N + 1) - tie_sum / (N * (N - 1)))
    if var <= 0:
        raise ValueError("all observations tied; the test is degenerate.")
    mean = m * (N + 1) / 2.0
    z = (W - mean) / np.sqrt(var)
    return RichResult(
        payload={
            "W": W, "mean": mean, "var_corrected": float(var), "z": float(z),
            "p_two_sided": float(2 * stats.norm.sf(abs(z))),
            "tie_sum": tie_sum, "m": m, "n": n,
            "method": "Wilcoxon rank-sum, tie-corrected variance (Gibbons Ch. 8.2)",
        }
    )


def cheatsheet():
    return "gb821t: Var(W) = Var(U); W = U + m(m+1)/2"
