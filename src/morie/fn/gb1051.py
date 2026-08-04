# morie.fn -- function file (rootcoder007/morie)
"""General k-sample rank statistic with arbitrary scores."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['krankstat', 'gibbons_k_rank_alt']


def krankstat(samples, scores=None):
    """Q from eq. (10.5.1) and its chi-square standardisation.

    Section 10.5 (book pp. 362-363).  Replacing the ranks by any
    monotone score function g gives

    .. math:: Q = \\sum_{j=1}^{k}\\frac{\\left[\\sum_i
        g(r_j(X_i)) - n_j \\bar a\\right]^2}{n_j},
        \\qquad \\bar a = \\frac{1}{N}\\sum_{i=1}^{N} a_i,

    and the book states that

    .. math:: \\frac{(N-1)Q}{\\sum_i (a_i - \\bar a)^2}

    tends to chi-square with k - 1 degrees of freedom.  With
    g(r) = r this is the Kruskal-Wallis statistic; the Terry and van
    der Waerden analogues named on the same page follow from the
    corresponding score vectors.

    Parameters
    ----------
    samples : sequence of sequence of float
        The k samples.
    scores : sequence of float, optional
        Scores a_1, ..., a_N indexed by pooled rank; defaults to the
        ranks themselves (Kruskal-Wallis).

    Returns
    -------
    RichResult
        keys ``q``, ``statistic`` (the standardised chi-square form),
        ``df``, ``p_value``, ``abar``, ``score_sums``, ``k``, ``n``,
        ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 10.5, eqs. (10.5.1)-(10.5.2),
    pp. 362-363.
    """
    ss = [[float(v) for v in s] for s in samples]
    k = len(ss)
    if k < 2:
        raise ValueError("need at least 2 samples.")
    flat = [(v, i) for i, s in enumerate(ss) for v in s]
    flat.sort(key=lambda p: p[0])
    nn = len(flat)
    if scores is None:
        a = [float(i + 1) for i in range(nn)]
    else:
        a = [float(v) for v in scores]
        if len(a) != nn:
            raise ValueError("scores must have length N.")
    abar = sum(a) / nn
    sums = [0.0] * k
    for idx, (_, grp) in enumerate(flat):
        sums[grp] += a[idx]
    ns = [len(s) for s in ss]
    q = sum((sums[j] - ns[j] * abar) ** 2 / ns[j] for j in range(k))
    ssq = sum((v - abar) ** 2 for v in a)
    stat = (nn - 1.0) * q / ssq if ssq > 0 else float("nan")
    return RichResult(
        payload={
            "q": float(q),
            "statistic": float(stat),
            "df": int(k - 1),
            "p_value": float(stats.chi2.sf(stat, k - 1)),
            "abar": float(abar),
            "score_sums": sums,
            "k": int(k),
            "n": int(nn),
            "method": "general k-sample rank statistic, eq. (10.5.1)",
        }
    )


gibbons_k_rank_alt = krankstat
