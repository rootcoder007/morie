# SPDX-License-Identifier: AGPL-3.0-or-later
"""Binary segmentation changepoint search."""

import math

from . import _array_core as np

from ._richresult import RichResult
from .pelt import _cost_tables, _make_cost

__all__ = ["binseg", "binary_segmentation"]


def binseg(x, K, cost="mean", penalty=0.0, min_seglen=1):
    """
    Binary segmentation for multiple changepoints.

    Iteratively applies the single-changepoint test of Killick,
    Fearnhead & Eckley (2012), eq (2): a split at tau inside a segment
    is accepted when C(y_{1:tau}) + C(y_{(tau+1):n}) + beta <
    C(y_{1:n}); the segment/location pair chosen at each step is the
    one with the largest cost reduction, and the procedure recurses on
    the two halves (their Sec. 2.1) until K changepoints are found or
    no split satisfies eq (2).

    Parameters
    ----------
    x : array-like
        Univariate series.
    K : int
        Maximum number of changepoints.
    cost : str
        "mean" or "meanvar" (see pelt).
    penalty : float
        beta in eq (2). Default 0.0 (pure forced-K search).
    min_seglen : int
        Minimum segment length.

    Returns
    -------
    result : RichResult
        Keys: changepoints (sorted, 1-based last index of segment),
        order (locations in detection order), improvements (cost
        reduction minus penalty at each accepted split), n_changepoints,
        segment_means.

    References
    ----------
    Killick, R., Fearnhead, P. and Eckley, I. A. (2012), JASA 107(500),
    1590-1598, Sec. 2.1 eq (2) (arXiv:1101.1438); method originally due
    to Scott, A. J. and Knott, M. (1974), "A cluster analysis method
    for grouping means in the analysis of variance", Biometrics 30(3),
    507-512, as attributed by Killick et al. Sec. 1.
    Source PDF: /run/media/rootcoder/WD_BLACK/library/pdf/fetched-wave3/
    killick-fearnhead-eckley-2012-pelt-optimal-changepoint-linear-cost.pdf
    """
    xv = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(xv)
    K = int(K)
    if n < 2 * min_seglen:
        raise ValueError("series too short")
    xs = [float(v) for v in xv]
    cs, css = _cost_tables(xs)
    C = _make_cost(cs, css, cost)

    def best_split(a, b):
        # best tau strictly inside [a, b) with both parts >= min_seglen
        best_gain = -math.inf
        best_tau = -1
        base = C(a, b)
        for tau in range(a + min_seglen, b - min_seglen + 1):
            g = base - (C(a, tau) + C(tau, b)) - penalty
            if g > best_gain:
                best_gain = g
                best_tau = tau
        return best_tau, best_gain

    segments = [(0, n)]
    order = []
    gains = []
    while len(order) < K:
        cand = None
        for (a, b) in segments:
            if b - a < 2 * min_seglen:
                continue
            tau, g = best_split(a, b)
            if tau > 0 and (cand is None or g > cand[2]):
                cand = (a, b, g, tau)
        if cand is None or cand[2] <= 0.0:
            break
        a, b, g, tau = cand
        order.append(tau)
        gains.append(float(g))
        segments.remove((a, b))
        segments.extend([(a, tau), (tau, b)])
    taus = sorted(order)
    bounds = [0] + taus + [n]
    seg_means = [float(np.mean(np.asarray(xs[a:b])))
                 for a, b in zip(bounds[:-1], bounds[1:])]
    return RichResult(payload={
        "changepoints": taus,
        "order": list(order),
        "improvements": gains,
        "n_changepoints": len(taus),
        "segment_means": seg_means,
        "estimate": taus,
        "n": n,
        "method": "Binary segmentation (Scott-Knott 1974; Killick et al. 2012 Sec. 2.1)",
    })


def binary_segmentation(x, K, cost="mean", penalty=0.0, min_seglen=1):
    """Alias for binseg (original stub export name)."""
    return binseg(x, K, cost=cost, penalty=penalty, min_seglen=min_seglen)


def cheatsheet():
    return "binseg(x, K) -> up to K changepoints by greedy binary segmentation"
