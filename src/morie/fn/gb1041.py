# morie.fn -- function file (rootcoder007/morie)
"""Kruskal-Wallis one-way ANOVA by ranks."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['kwh', 'gibbons_kruskal_wallis']


def kwh(samples, correct=True):
    """H statistic with the tie correction of eq. (10.4.5).

    Section 10.4 (book pp. 354-358).  Pool and rank all N
    observations; with R_i the rank sum of sample i, eq. (10.4.7)
    gives the computing form

    .. math:: H = \\frac{12}{N(N+1)}\\sum_{i=1}^{k}
        \\frac{R_i^2}{n_i} - 3(N+1),

    which is eq. (10.4.2) rearranged.  Ties handled by midranks divide
    H by 1 - sum t(t^2-1)/[N(N^2-1)], eq. (10.4.5).  H is
    asymptotically chi-square on k - 1 degrees of freedom.

    The book's Example 10.4.1 (rank sums 260, 122, 90, 348 with
    n_i = 10, N = 40) gives H = 31.89.

    Parameters
    ----------
    samples : sequence of sequence of float
        The k samples, k >= 2.
    correct : bool, optional
        Apply the tie correction (default True).

    Returns
    -------
    RichResult
        keys ``statistic`` (H, corrected), ``h_raw``, ``correction``,
        ``df``, ``p_value``, ``rank_sums``, ``rank_means``, ``k``,
        ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 10.4, eqs. (10.4.2), (10.4.5),
    (10.4.7), pp. 354-358 (Kruskal and Wallis, 1952).
    """
    ss = [[float(v) for v in s] for s in samples]
    k = len(ss)
    if k < 2:
        raise ValueError("need at least 2 samples.")
    if any(len(s) < 1 for s in ss):
        raise ValueError("every sample must be non-empty.")
    flat = [(v, i) for i, s in enumerate(ss) for v in s]
    flat.sort(key=lambda p: p[0])
    nn = len(flat)
    ranks = [0.0] * nn
    ties = []
    i = 0
    while i < nn:
        j = i
        while j + 1 < nn and flat[j + 1][0] == flat[i][0]:
            j += 1
        mid = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[t] = mid
        if j > i:
            ties.append(j - i + 1)
        i = j + 1
    rs = [0.0] * k
    for idx, (_, grp) in enumerate(flat):
        rs[grp] += ranks[idx]
    ns = [len(s) for s in ss]
    h = 12.0 / (nn * (nn + 1.0)) * sum(
        rs[i] ** 2 / ns[i] for i in range(k)
    ) - 3.0 * (nn + 1.0)
    corr = 1.0
    if correct and ties:
        corr = 1.0 - sum(
            t * (t * t - 1.0) for t in ties
        ) / (nn * (float(nn) ** 2 - 1.0))
    hc = h / corr if corr > 0 else float("nan")
    return RichResult(
        payload={
            "statistic": float(hc),
            "h_raw": float(h),
            "correction": float(corr),
            "df": int(k - 1),
            "p_value": float(stats.chi2.sf(hc, k - 1)),
            "rank_sums": rs,
            "rank_means": [rs[i] / ns[i] for i in range(k)],
            "k": int(k),
            "n": int(nn),
            "method": "Kruskal-Wallis H, eqs. (10.4.2)/(10.4.7)",
        }
    )


gibbons_kruskal_wallis = kwh
