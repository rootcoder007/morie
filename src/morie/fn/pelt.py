# SPDX-License-Identifier: AGPL-3.0-or-later
"""PELT (pruned exact linear time) changepoint detection."""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["pelt"]


def _cost_tables(x):
    n = len(x)
    cs = [0.0] * (n + 1)
    css = [0.0] * (n + 1)
    for i in range(n):
        cs[i + 1] = cs[i] + float(x[i])
        css[i + 1] = css[i] + float(x[i]) * float(x[i])
    return cs, css


def _make_cost(cs, css, cost):
    # Segment cost C(y_{(a+1):b}) on the 0-based half-open block [a, b).
    # "mean": Normal change in mean, unit variance -- twice negative
    #   log-likelihood up to a constant: sum of squared deviations from
    #   the segment mean (Killick, Fearnhead & Eckley 2012, Sec. 2,
    #   below eq (1); Horvath 1993 convention).
    # "meanvar": Normal change in mean and variance -- twice negative
    #   log-likelihood n_l (log 2 pi + log sigma2_hat + 1) with the
    #   biased MLE sigma2_hat (same convention as the changepoint R
    #   package used for anchoring).
    log2pi = math.log(2.0 * math.pi)

    def C(a, b):
        nl = b - a
        s = cs[b] - cs[a]
        ssdev = css[b] - css[a] - s * s / nl
        if cost == "mean":
            return ssdev
        if cost == "meanvar":
            sig = ssdev / nl
            if sig <= 1e-300:
                sig = 1e-300
            return nl * (log2pi + math.log(sig) + 1.0)
        raise ValueError("cost must be 'mean' or 'meanvar'")

    return C


def _pelt_core(x, cost, penalty, min_seglen=1):
    n = len(x)
    cs, css = _cost_tables(x)
    C = _make_cost(cs, css, cost)
    beta = penalty
    # Algorithm 2 of Killick et al. (2012): F(0) = -beta, R_1 = {0};
    # F(t*) = min_{tau in R} [F(tau) + C(y_{(tau+1):t*}) + beta];
    # prune with K = 0 (Theorem 3.1, eq (4)-(5): K = 0 for a
    # log-likelihood cost).
    F = [0.0] * (n + 1)
    F[0] = -beta
    cp = [0] * (n + 1)
    Rset = [0]
    K = 0.0
    for t in range(min_seglen, n + 1):
        best = math.inf
        barg = 0
        for tau in Rset:
            if t - tau < min_seglen:
                continue
            v = F[tau] + C(tau, t) + beta
            if v < best:
                best = v
                barg = tau
        F[t] = best
        cp[t] = barg
        # pruning: keep tau with F(tau) + C(tau, t) + K <= F(t)
        Rset = [tau for tau in Rset
                if t - tau < min_seglen or F[tau] + C(tau, t) + K <= F[t]]
        Rset.append(t)
    # backtrack
    taus = []
    t = n
    while cp[t] > 0:
        taus.append(cp[t])
        t = cp[t]
    taus.reverse()
    return taus, F[n]


def pelt(x, cost="mean", penalty=None, min_seglen=1):
    """
    PELT -- Pruned Exact Linear Time changepoint detection.

    Minimises sum_{i=1}^{m+1} [ C(y_{(tau_{i-1}+1):tau_i}) + beta ]
    (Killick, Fearnhead & Eckley 2012, eq (1) with f(m) = m and eq (3))
    by the Optimal Partitioning recursion
    F(s) = min_t { F(t) + C(y_{(t+1):s}) + beta } (their Sec. 2.2,
    Algorithm 1) with the PELT pruning of Theorem 3.1 / Algorithm 2:
    t is discarded once F(t) + C(y_{(t+1):s}) + K > F(s), with K = 0
    for the (twice negative) log-likelihood costs used here.

    Parameters
    ----------
    x : array-like
        Univariate series y_1..y_n.
    cost : str
        "mean" (Normal change in mean, unit variance: segment cost =
        sum of squared deviations) or "meanvar" (Normal change in mean
        and variance: n_l (log 2 pi + log sigma2_hat + 1)).
    penalty : float, optional
        beta. Default: p log n (SIC/BIC, Killick et al. 2012 Sec. 2)
        with p = 1 for "mean", p = 2 for "meanvar".
    min_seglen : int
        Minimum segment length (paper Sec. 2, last paragraph).

    Returns
    -------
    result : RichResult
        Keys: changepoints (1-based positions tau_i = last index of
        each segment except the final one), n_changepoints, objective
        (F(n)), penalty, segment_means.

    References
    ----------
    Killick, R., Fearnhead, P. and Eckley, I. A. (2012), "Optimal
    detection of changepoints with a linear computational cost",
    Journal of the American Statistical Association 107(500),
    1590-1598 (arXiv:1101.1438). Equations (1)-(5), Algorithms 1-2.
    Source PDF: /run/media/rootcoder/WD_BLACK/library/pdf/fetched-wave3/
    killick-fearnhead-eckley-2012-pelt-optimal-changepoint-linear-cost.pdf
    """
    xv = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(xv)
    if cost == "meanvar" and min_seglen < 2:
        # sigma2_hat = 0 on singleton segments makes the likelihood
        # unbounded; a minimum segment length of 2 is required (cf.
        # Killick et al. 2012 Sec. 2 on minimum segment lengths, and
        # the changepoint package convention).
        min_seglen = 2
    if n < 2 * min_seglen:
        raise ValueError("series too short")
    if penalty is None:
        p = 1.0 if cost == "mean" else 2.0
        penalty = p * math.log(n)
    xs = [float(v) for v in xv]
    taus, Fn = _pelt_core(xs, cost, float(penalty), min_seglen)
    bounds = [0] + taus + [n]
    seg_means = [float(np.mean(np.asarray(xs[a:b])))
                 for a, b in zip(bounds[:-1], bounds[1:])]
    return RichResult(payload={
        "changepoints": list(taus),
        "n_changepoints": len(taus),
        "objective": float(Fn),
        "penalty": float(penalty),
        "segment_means": seg_means,
        "estimate": list(taus),
        "n": n,
        "method": "PELT (Killick-Fearnhead-Eckley 2012)",
    })


def cheatsheet():
    return "pelt(x, cost='mean'|'meanvar', penalty) -> exact penalised changepoints via pruned DP"
