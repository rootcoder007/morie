"""Bootstrap-t all-pairs comparison of trimmed means (FWE control through Tmax).

Wilcox (2017), Modern Statistics for the Social and Behavioral Sciences, Sec 12.1.13 (eq 8.12 d_j).
"""

import math

from ._richresult import RichResult
from ._rng import random_uniform
from .trimse import _tmean
from .yuen import _yuen_d

__all__ = ["linconbt"]


def linconbt(groups, tr=0.2, alpha=0.05, B=599, seed=0):
    """Compare all pairs of trimmed means with a bootstrap-t critical value for the maximum.

    Each bootstrap round resamples every group (morie Philox stream j for group
    j), computes trimmed means and Yuen's squared standard errors d*_j, and
    records T*max = max_{j<k} |(X*_tj - X*_tk) - (X_tj - X_tk)| / sqrt(d*_j + d*_k).
    With u = (1 - alpha) B rounded, the intervals are
    (X_tj - X_tk) +- T*max_(u) sqrt(d_j + d_k); H0 is rejected when
    |X_tj - X_tk| / sqrt(d_j + d_k) >= T*max_(u).

    Parameters
    ----------
    groups : list of sequences
    tr, alpha : float
    B : int
    seed : int

    Returns
    -------
    RichResult
        Keys: crit, comparisons (j, k, diff, lower, upper, test, reject), tmax (sorted).

    References
    ----------
    Wilcox, R. R. (2017). Modern Statistics for the Social and Behavioral
    Sciences (2nd ed.). CRC Press. Sec 12.1.13.

    Examples
    --------
    >>> r = linconbt([[1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9], [5, 6, 7, 8, 9, 10, 11, 12]], B=99)
    >>> len(r["comparisons"])
    3
    """
    gs = [[float(v) for v in g] for g in groups]
    J = len(gs)
    if J < 2 or any(len(g) - 2 * math.floor(tr * len(g)) < 2 for g in gs):
        raise ValueError("need at least two groups with 2+ observations left after trimming")
    est = [_tmean(g, tr) for g in gs]
    d = [_yuen_d(g, tr)[0] for g in gs]
    us = [random_uniform(B * len(g), seed=seed, stream=j) for j, g in enumerate(gs)]
    tmax = []
    for b in range(B):
        bs = [[g[math.floor(u[b * len(g) + i] * len(g))] for i in range(len(g))] for g, u in zip(gs, us)]
        eb = [_tmean(g, tr) for g in bs]
        db = [_yuen_d(g, tr)[0] for g in bs]
        tmax.append(
            max(
                abs((eb[j] - eb[k]) - (est[j] - est[k])) / math.sqrt(db[j] + db[k])
                for j in range(J)
                for k in range(j + 1, J)
            )
        )
    tmax.sort()
    crit = tmax[round((1 - alpha) * B) - 1]
    comps = []
    for j in range(J):
        for k in range(j + 1, J):
            se = math.sqrt(d[j] + d[k])
            diff = est[j] - est[k]
            comps.append(
                {
                    "j": j,
                    "k": k,
                    "diff": diff,
                    "lower": diff - crit * se,
                    "upper": diff + crit * se,
                    "test": abs(diff) / se,
                    "reject": abs(diff) / se >= crit,
                }
            )
    return RichResult(
        title="Bootstrap-t all-pairs comparison of trimmed means",
        summary_lines=[("crit", crit)],
        payload={"crit": crit, "comparisons": comps, "tmax": tmax},
    )


def cheatsheet():
    return "linconbt: bootstrap-t Tmax all-pairs trimmed-mean comparisons. Wilcox (2017) Sec 12.1.13."
