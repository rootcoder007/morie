# morie.fn -- function file (rootcoder007/morie)
"""Kruskal-Wallis with the tie correction."""

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["gibbons_kw_ties"]


def gibbons_kw_ties(groups):
    r"""Kruskal-Wallis H with the tie-corrected denominator (Gibbons
    Ch. 10.4):

    .. math:: H_{adj} = \frac{H}{1 - \sum t(t^2 - 1)/(N^3 - N)}.

    The correction DIVIDES by a quantity below one, so ties always
    raise H -- the uncorrected statistic is conservative, same
    direction as in the two-sample tests.

    Parameters
    ----------
    groups : sequence of array-like
        k >= 2 samples.

    Returns
    -------
    RichResult
        keys: ``H``, ``H_uncorrected``, ``correction``, ``df``,
        ``p_value``, ``tie_sum``, ``k``, ``N``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 10.4.
    """
    gs = [np.asarray(g, dtype=float).ravel() for g in groups]
    k = len(gs)
    if k < 2:
        raise ValueError("need at least 2 groups.")
    if any(g.size < 1 for g in gs):
        raise ValueError("every group must be non-empty.")
    combined = np.concatenate(gs)
    N = combined.size
    if N < 3:
        raise ValueError("need at least 3 observations in total.")
    ranks = stats.rankdata(combined)
    H = 0.0
    pos = 0
    for g in gs:
        r = ranks[pos : pos + g.size]
        H += r.sum() ** 2 / g.size
        pos += g.size
    H = 12.0 / (N * (N + 1)) * H - 3.0 * (N + 1)
    _, counts = np.unique(combined, return_counts=True)
    tie_sum = float(np.sum(counts * (counts**2 - 1)))
    corr = 1.0 - tie_sum / (N**3 - N)
    if corr <= 0:
        raise ValueError("all observations tied; H is degenerate.")
    H_adj = H / corr
    return RichResult(
        payload={
            "H": float(H_adj), "H_uncorrected": float(H), "correction": float(corr),
            "df": int(k - 1), "p_value": float(stats.chi2.sf(H_adj, k - 1)),
            "tie_sum": tie_sum, "k": k, "N": int(N),
            "method": "Kruskal-Wallis H / (1 - sum t(t^2-1)/(N^3-N)) (Ch. 10.4)",
        }
    )


def cheatsheet():
    return "gb1041t: tie correction divides by < 1, so H only goes UP"


# compact alias per ledger/NAMING.md
gibbonskwties = gibbons_kw_ties
