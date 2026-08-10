# SPDX-License-Identifier: AGPL-3.0-or-later
"""E-divisive multiple changepoint estimation (energy distance)."""

import math

from . import _array_core as np

from ._richresult import RichResult
from ._rng import random_uniform

__all__ = ["e_div", "e_divisive", "edivisive"]


def _pairwise_alpha(z, alpha):
    # D[i][j] = |z_i - z_j|^alpha (Euclidean norm for multivariate rows)
    n = len(z)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        zi = z[i]
        for j in range(i + 1, n):
            zj = z[j]
            if isinstance(zi, list):
                s = 0.0
                for a, b in zip(zi, zj):
                    s += (a - b) * (a - b)
                d = math.sqrt(s)
            else:
                d = abs(zi - zj)
            v = d ** alpha
            D[i][j] = v
            D[j][i] = v
    return D


def _prefix2d(D):
    n = len(D)
    P = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n):
        row = D[i]
        Pi = P[i]
        Pi1 = P[i + 1]
        for j in range(n):
            Pi1[j + 1] = Pi1[j] + Pi[j + 1] - Pi[j] + row[j]
    return P


def _block(P, a1, b1, a2, b2):
    # sum of D[i][j] for i in [a1,b1), j in [a2,b2)
    return P[b1][b2] - P[a1][b2] - P[b1][a2] + P[a1][a2]


def _qhat(P, a, tau, kappa):
    # Ehat(X, Y; alpha) of Matteson & James (2014) eq (5) with
    # X = z[a:tau], Y = z[tau:kappa]; Qhat = (mn/(m+n)) Ehat, eq (6).
    n1 = tau - a
    m1 = kappa - tau
    between = _block(P, a, tau, tau, kappa)
    withinX = _block(P, a, tau, a, tau) / 2.0
    withinY = _block(P, tau, kappa, tau, kappa) / 2.0
    e = (2.0 / (n1 * m1)) * between \
        - withinX / (n1 * (n1 - 1) / 2.0) \
        - withinY / (m1 * (m1 - 1) / 2.0)
    return (n1 * m1 / float(n1 + m1)) * e


def _best_split(P, a, b, min_size):
    # eq (7): (tau, kappa) = argmax Qhat(X_tau, Y_tau(kappa)); the
    # scan is row-major in (tau, kappa) with strict > so ties resolve
    # identically in both language arms.
    best = (-math.inf, -1, -1)
    for tau in range(a + min_size, b - min_size + 1):
        for kappa in range(tau + min_size, b + 1):
            q = _qhat(P, a, tau, kappa)
            if q > best[0]:
                best = (q, tau, kappa)
    return best


def _shuffle_within(order, clusters, us, pos):
    # Fisher-Yates within each cluster, consuming uniforms us[pos...]
    # identically in the R arm.
    for (a, b) in clusters:
        L = b - a
        for i in range(L - 1, 0, -1):
            j = int(us[pos] * (i + 1))
            if j > i:
                j = i
            pos += 1
            order[a + i], order[a + j] = order[a + j], order[a + i]
    return pos


def e_div(x, sig=0.05, R=199, alpha=1.0, min_size=2, max_cp=None,
          seed=20260809):
    """
    E-divisive hierarchical multiple changepoint estimation.

    Implements Matteson & James (2014): the empirical divergence
    Ehat(X_n, Y_m; alpha) of eq (5) (energy statistic with between- and
    within-sample Euclidean-distance terms), the scaled statistic
    Qhat = (mn/(m+n)) Ehat of eq (6), the single-changepoint search
    (tau_hat, kappa_hat) = argmax Qhat of eq (7), hierarchical
    application within existing clusters (Sec. 2.3, test statistic
    q_hat_k of eq (8)), and the permutation significance test of
    Sec. 2.4: observations are permuted within each existing cluster,
    the estimation is reapplied, and the approximate p-value is
    #{r : q_hat^(r) >= q_hat} / (R + 1); the procedure stops when
    p > sig.

    Permutations use the native Philox4x32 stream (bit-identical in
    the R arm), so results are exactly reproducible cross-language.

    Parameters
    ----------
    x : array-like
        Series (1-d) or matrix with observations in rows.
    sig : float
        Stopping significance level p0 (paper uses 0.05).
    R : int
        Number of random permutations (paper uses 499).
    alpha : float
        Index in (0, 2) of the divergence (paper eq (4); default 1).
    min_size : int
        Minimum segment size (>= 2 so the U-statistic in eq (5) is
        defined).
    max_cp : int, optional
        Optional cap on the number of changepoints.
    seed : int
        Philox seed for the permutation streams.

    Returns
    -------
    result : RichResult
        Keys: changepoints (1-based last index of each left segment,
        in detection order), changepoints_sorted, p_values, q_stats,
        n_changepoints.

    References
    ----------
    Matteson, D. S. and James, N. A. (2014), "A nonparametric approach
    for multiple change point analysis of multivariate data", Journal
    of the American Statistical Association 109(505), 334-345
    (arXiv:1306.4933). Equations (4)-(8), Sections 2.1-2.4.
    Divergence measure: Szekely, G. J. and Rizzo, M. L. (2005),
    "Hierarchical clustering via joint between-within distances",
    as cited therein.
    Source PDF: /run/media/rootcoder/WD_BLACK/library/pdf/fetched-wave3/
    matteson-james-2014-edivisive-nonparametric-changepoint.pdf
    """
    xv = np.asarray(x, dtype=float)
    if xv.ndim == 1:
        z = [float(v) for v in xv]
    else:
        z = [[float(v) for v in row] for row in xv]
    n = len(z)
    if n < 2 * min_size:
        raise ValueError("series too short")
    if not (0.0 < alpha < 2.0):
        raise ValueError("alpha must be in (0, 2)")
    order = list(range(n))
    cps = []
    pvals = []
    qstats = []
    clusters_of = lambda taus: [(a, b) for a, b in
                                zip([0] + sorted(taus), sorted(taus) + [n])]
    D = _pairwise_alpha(z, alpha)
    P = _prefix2d(D)
    while True:
        if max_cp is not None and len(cps) >= max_cp:
            break
        clusters = clusters_of(cps)
        # observed statistic: best split over all clusters (Sec. 2.3)
        best = (-math.inf, -1, -1)
        for (a, b) in clusters:
            if b - a >= 2 * min_size:
                q, tau, kappa = _best_split(P, a, b, min_size)
                if q > best[0]:
                    best = (q, tau, kappa)
        qhat, tau_hat, _ = best
        if tau_hat < 0:
            break
        # permutation test (Sec. 2.4)
        # uniforms per permutation: one per within-cluster swap
        needed = sum((b - a - 1) for (a, b) in clusters if b - a > 1)
        count_ge = 0
        for r in range(R):
            us = random_uniform(needed, seed=seed, stream=r + 1)
            perm = list(range(n))
            _shuffle_within(perm, clusters, us, 0)
            zp = [z[perm[i]] for i in range(n)]
            Dp = _pairwise_alpha(zp, alpha)
            Pp = _prefix2d(Dp)
            bq = -math.inf
            for (a, b) in clusters:
                if b - a >= 2 * min_size:
                    q, _, _ = _best_split(Pp, a, b, min_size)
                    if q > bq:
                        bq = q
            if bq >= qhat:
                count_ge += 1
        p = count_ge / float(R + 1)
        pvals.append(p)
        qstats.append(float(qhat))
        if p > sig:
            break
        cps.append(tau_hat)
    return RichResult(payload={
        "changepoints": list(cps),
        "changepoints_sorted": sorted(cps),
        "p_values": pvals,
        "q_stats": qstats,
        "n_changepoints": len(cps),
        "estimate": sorted(cps),
        "n": n,
        "method": "E-divisive (Matteson-James 2014)",
    })


def e_divisive(x, sig=0.05, **kw):
    """Alias for e_div (original stub export name)."""
    return e_div(x, sig=sig, **kw)


edivisive = e_divisive


def cheatsheet():
    return "e_div(x, sig) -> energy-distance hierarchical changepoints with permutation stopping"
