# morie.fn -- function file (rootcoder007/morie)
"""Space-time regression and state models: geographically and temporally weighted regression
(GTWR), multiscale GWR by backfitting over space(-time) coordinates, and the Gaussian hidden
Markov model fitted by Baum-Welch with Viterbi decoding for sets of site series."""

from __future__ import annotations

import math

from ._qpcore import solve, ssum
from ._richresult import RichResult

__all__ = ["gtwr_fit", "mgwr_backfit", "gaussian_hmm"]


def _kernel(d, bw, kernel):
    if kernel == "gaussian":
        return math.exp(-0.5 * (d / bw) ** 2)
    if kernel == "exponential":
        return math.exp(-d / bw)
    return (1.0 - (d / bw) ** 2) ** 2 if d < bw else 0.0


def _st_dist(ci, cj, ti, tj, lam, mu, distance, ksi):
    ds = math.hypot(ci[0] - cj[0], ci[1] - cj[1])
    dt = abs(ti - tj)
    if distance == "fotheringham":
        return lam * ds + (1 - lam) * dt + 2 * math.sqrt(lam * (1 - lam) * ds * dt) * math.cos(ksi)
    return math.sqrt(lam * ds * ds + mu * dt * dt)


def gtwr_fit(
    y,
    X,
    coords,
    times,
    bandwidth: float,
    *,
    lam: float = 1.0,
    mu: float = 1.0,
    kernel: str = "bisquare",
    distance: str = "huang",
    ksi: float = 0.0,
    past_only: bool = False,
) -> RichResult:
    r"""Geographically and temporally weighted regression (GTWR).

    At each observation ``i`` the local coefficients solve the weighted least
    squares ``beta_i = (X'W_i X)^(-1) X'W_i y`` (an intercept is added) with
    kernel weights of the space-time distance: Huang, Wu and Barry (2010)
    ``d = sqrt(lam d_S^2 + mu d_T^2)`` (default) or, with
    ``distance="fotheringham"``, ``d = lam d_S + (1 - lam) d_T + 2 sqrt(lam (1 - lam) d_S d_T) cos(ksi)``
    as ``GWmodel::gtwr``; with ``past_only`` only observations at or before the
    focal time get weight (GWmodel's convention). Kernels: bisquare ``(1 - (d/b)^2)^2`` for
    ``d < b``, gaussian ``exp(-(d/b)^2 / 2)``, exponential ``exp(-d/b)``.

    References
    ----------
    Huang, B., Wu, B. and Barry, M. (2010). Geographically and temporally
    weighted regression for modeling spatio-temporal variation in house
    prices. IJGIS 24, 383-401. Fotheringham, A. S., Crespo, R. and Yao, J.
    (2015). Geographical and temporal weighted regression (GTWR).
    Geographical Analysis 47, 431-452.

    Examples
    --------
    >>> xy = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (2, 2)]
    >>> r = gtwr_fit([1, 2, 1.5, 3, 3.2, 4], [[0.1], [0.5], [0.3], [0.9], [1.1], [1.4]], xy, [0, 0, 1, 1, 2, 2], 5.0)
    >>> len(r.beta), len(r.beta[0])
    (6, 2)
    """
    ys = [float(v) for v in y]
    n = len(ys)
    Xr = [[1.0] + [float(v) for v in r] for r in X]
    p = len(Xr[0])
    betas, fitted = [], []
    for i in range(n):
        w = [
            0.0
            if past_only and times[j] > times[i]
            else _kernel(_st_dist(coords[i], coords[j], times[i], times[j], lam, mu, distance, ksi), bandwidth, kernel)
            for j in range(n)
        ]
        G = [[ssum(w[j] * Xr[j][a] * Xr[j][b] for j in range(n)) for b in range(p)] for a in range(p)]
        h = [ssum(w[j] * Xr[j][a] * ys[j] for j in range(n)) for a in range(p)]
        b = solve(G, h)
        betas.append(b)
        fitted.append(ssum(Xr[i][a] * b[a] for a in range(p)))
    res = [a - b for a, b in zip(ys, fitted)]
    return RichResult(payload={"beta": betas, "fitted": fitted, "residuals": res, "rss": ssum(v * v for v in res)})


def mgwr_backfit(
    y,
    X,
    coords,
    bandwidths,
    *,
    kernel: str = "bisquare",
    center: bool = True,
    threshold: float = 1e-10,
    max_iter: int = 5000,
) -> RichResult:
    r"""Multiscale geographically weighted regression by backfitting with fixed per-covariate bandwidths.

    ``y = sum_j beta_j(u) x_j + e`` with an intercept first; the non-intercept
    columns are centred when ``center`` (as ``GWmodel::gwr.multiscale``'s
    ``predictor.centered``). Starting from OLS, each smooth is refitted as a
    one-covariate GWR of the partial residual ``y - sum_{k != j} f_k`` with
    its own bandwidth ``bandwidths[j]`` until
    ``sqrt(|RSS_new - RSS_old| / RSS_new) < threshold`` (the dCVR criterion).
    With centring the reported local intercepts refer to the uncentred
    predictors, as GWmodel's. ``coords`` may carry any number of dimensions, e.g. ``(x, y, c t)`` for
    space-time variation.

    References
    ----------
    Fotheringham, A. S., Yang, W. and Kang, W. (2017). Multiscale
    geographically weighted regression (MGWR). Annals AAG 107, 1247-1265.
    Lu, B., Brunsdon, C., Charlton, M. and Harris, P. (2017). Geographically
    weighted regression with parameter-specific distance metrics. IJGIS 31, 982-998.

    Examples
    --------
    >>> xy = [(i % 4, i // 4) for i in range(12)]
    >>> x = [[math.sin(i)] for i in range(12)]
    >>> r = mgwr_backfit([1 + 2 * v[0] for v in x], x, xy, [10.0, 10.0])
    >>> round(r.beta[0][1], 8)
    2.0
    """
    ys = [float(v) for v in y]
    n = len(ys)
    cols = [[1.0] * n] + [[float(r[k]) for r in X] for k in range(len(X[0]))]
    means = [ssum(c) / n for c in cols[1:]]
    if center:
        cols = [cols[0]] + [[v - m for v in c] for c, m in zip(cols[1:], means)]
    p = len(cols)
    D = [[math.sqrt(ssum((a - b) ** 2 for a, b in zip(coords[i], coords[j]))) for j in range(n)] for i in range(n)]
    Wt = [[[_kernel(D[i][j], bandwidths[k], kernel) for j in range(n)] for i in range(n)] for k in range(p)]
    G = [[ssum(cols[a][i] * cols[b][i] for i in range(n)) for b in range(p)] for a in range(p)]
    b0 = solve(G, [ssum(cols[a][i] * ys[i] for i in range(n)) for a in range(p)])
    beta = [[b0[k]] * n for k in range(p)]
    f = [[cols[k][i] * beta[k][i] for i in range(n)] for k in range(p)]
    resid = [ys[i] - ssum(f[k][i] for k in range(p)) for i in range(n)]
    rss0 = ssum(v * v for v in resid)
    it = 0
    crit = math.inf
    while it < max_iter and crit > threshold:
        it += 1
        for k in range(p):
            yk = [resid[i] + f[k][i] for i in range(n)]
            x = cols[k]
            W = Wt[k]
            bk = [
                ssum(W[i][j] * x[j] * yk[j] for j in range(n)) / ssum(W[i][j] * x[j] * x[j] for j in range(n))
                for i in range(n)
            ]
            beta[k] = bk
            f[k] = [x[i] * bk[i] for i in range(n)]
            resid = [yk[i] - f[k][i] for i in range(n)]
        rss1 = ssum(v * v for v in resid)
        crit = math.sqrt(abs(rss1 - rss0) / rss1) if rss1 > 0 else 0.0
        rss0 = rss1
    if center:
        # report the local intercept for the uncentred predictors
        beta[0] = [beta[0][i] - ssum(beta[k][i] * means[k - 1] for k in range(1, p)) for i in range(n)]
    return RichResult(
        payload={
            "beta": [[beta[k][i] for k in range(p)] for i in range(n)],
            "fitted": [ys[i] - resid[i] for i in range(n)],
            "residuals": resid,
            "rss": rss0,
            "iterations": it,
        }
    )


def _q7(s, prob):
    h = (len(s) - 1) * prob
    lo = int(math.floor(h))
    hi = min(lo + 1, len(s) - 1)
    w = h - lo
    return (1.0 - w) * s[lo] + w * s[hi] if w > 0 else s[lo]


def _ldnorm(x, m, s):
    return -0.5 * ((x - m) / s) ** 2 - math.log(s) - 0.5 * math.log(2 * math.pi)


def gaussian_hmm(
    series, n_states: int, *, means=None, sds=None, trans=None, init=None, max_iter: int = 1000, tol: float = 1e-12
) -> RichResult:
    r"""Gaussian hidden Markov model by the Baum-Welch (EM) algorithm, with Viterbi decoding.

    ``series`` is one sequence or a list of sequences (e.g. one per site)
    sharing the parameters. Scaled forward-backward recursions give the
    state and transition posteriors; the M step updates the initial
    distribution (mean of the first-period posteriors), the transition
    matrix, and the state means and standard deviations (Rabiner 1989).
    Defaults start the means at the type 7 quantiles ``(k + 1/2)/K`` of the
    pooled data, the standard deviations at the pooled one, transitions at
    0.9 on the diagonal. Iterates until the log-likelihood gain is below ``tol``.

    References
    ----------
    Baum, L. E., Petrie, T., Soules, G. and Weiss, N. (1970). A maximization
    technique occurring in the statistical analysis of probabilistic
    functions of Markov chains. Ann. Math. Statist. 41, 164-171. Rabiner, L.
    R. (1989). A tutorial on hidden Markov models and selected applications in
    speech recognition. Proc. IEEE 77, 257-286.

    Examples
    --------
    >>> x = [0.1, -0.2, 0.0, 5.1, 4.9, 5.2, 0.2, -0.1]
    >>> r = gaussian_hmm(x, 2)
    >>> r.states[0]
    [0, 0, 0, 1, 1, 1, 0, 0]
    """
    seqs = (
        [list(map(float, series))]
        if not isinstance(series[0], (list, tuple))
        else [list(map(float, s)) for s in series]
    )
    K = n_states
    pooled = sorted(v for s in seqs for v in s)
    N = len(pooled)
    gm = ssum(pooled) / N
    gsd = math.sqrt(ssum((v - gm) ** 2 for v in pooled) / N)
    mu = [float(v) for v in means] if means is not None else [_q7(pooled, (k + 0.5) / K) for k in range(K)]
    sd = [float(v) for v in sds] if sds is not None else [gsd] * K
    A = (
        [list(map(float, r)) for r in trans]
        if trans is not None
        else [[0.9 if i == j else 0.1 / (K - 1) for j in range(K)] for i in range(K)]
    )
    pi = [float(v) for v in init] if init is not None else [1.0 / K] * K

    def fb(x):
        T = len(x)
        # densities scaled by their per-period maximum (added back to the log-likelihood)
        LB = [[_ldnorm(v, mu[k], sd[k]) for k in range(K)] for v in x]
        mx = [max(r) for r in LB]
        B = [[math.exp(r[k] - m) for k in range(K)] for r, m in zip(LB, mx)]
        al, c = [], []
        a = [pi[k] * B[0][k] for k in range(K)]
        s = ssum(a)
        al.append([v / s for v in a])
        c.append(s)
        for t in range(1, T):
            a = [B[t][k] * ssum(al[t - 1][i] * A[i][k] for i in range(K)) for k in range(K)]
            s = ssum(a)
            al.append([v / s for v in a])
            c.append(s)
        be = [[1.0] * K for _ in range(T)]
        for t in range(T - 2, -1, -1):
            be[t] = [ssum(A[k][j] * B[t + 1][j] * be[t + 1][j] for j in range(K)) / c[t + 1] for k in range(K)]
        return B, al, be, c, mx

    ll_old = -math.inf
    ll = -math.inf
    it = 0
    for _ in range(max_iter):
        it += 1
        num_pi = [0.0] * K
        num_A = [[0.0] * K for _ in range(K)]
        g_sum = [0.0] * K
        g_x = [0.0] * K
        g_xx = [0.0] * K
        ll = 0.0
        for x in seqs:
            B, al, be, c, mx = fb(x)
            ll += ssum(math.log(v) for v in c) + ssum(mx)
            T = len(x)
            for t in range(T):
                g = [al[t][k] * be[t][k] for k in range(K)]
                for k in range(K):
                    g_sum[k] += g[k]
                    g_x[k] += g[k] * x[t]
                    g_xx[k] += g[k] * x[t] * x[t]
                if t == 0:
                    for k in range(K):
                        num_pi[k] += g[k]
                if t > 0:
                    for i in range(K):
                        for j in range(K):
                            num_A[i][j] += al[t - 1][i] * A[i][j] * B[t][j] * be[t][j] / c[t]
        pi = [v / len(seqs) for v in num_pi]
        A = [[num_A[i][j] / ssum(num_A[i]) for j in range(K)] for i in range(K)]
        mu = [g_x[k] / g_sum[k] for k in range(K)]
        sd = [math.sqrt(max(g_xx[k] / g_sum[k] - mu[k] * mu[k], 0.0)) for k in range(K)]
        if ll - ll_old < tol:
            break
        ll_old = ll
    # final log-likelihood and Viterbi paths
    ll = 0.0
    states = []
    for x in seqs:
        _, _, _, c, mx = fb(x)
        ll += ssum(math.log(v) for v in c) + ssum(mx)
        T = len(x)
        d = [math.log(pi[k]) + _ldnorm(x[0], mu[k], sd[k]) if pi[k] > 0 else -math.inf for k in range(K)]
        bp = []
        for t in range(1, T):
            row, ptr = [], []
            for k in range(K):
                cand = [d[i] + (math.log(A[i][k]) if A[i][k] > 0 else -math.inf) for i in range(K)]
                i = max(range(K), key=lambda q: (cand[q], -q))
                row.append(cand[i] + _ldnorm(x[t], mu[k], sd[k]))
                ptr.append(i)
            d = row
            bp.append(ptr)
        path = [max(range(K), key=lambda q: (d[q], -q))]
        for ptr in reversed(bp):
            path.append(ptr[path[-1]])
        states.append(path[::-1])
    return RichResult(
        payload={"means": mu, "sds": sd, "trans": A, "init": pi, "loglik": ll, "states": states, "iterations": it}
    )


def cheatsheet() -> str:
    return "gtwr_fit / mgwr_backfit / gaussian_hmm -> space-time regression and hidden Markov state models."
