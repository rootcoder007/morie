# morie.fn -- function file (rootcoder007/morie)
"""MGWR Monte-Carlo stationarity test."""

import math

from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_uniform
from .mgwrfit import _backfit, _prep


def _var_betas(yv, Xm, D, bws, kernel, adaptive, threshold, max_iter):
    beta = _backfit(yv, Xm, D, bws, kernel, adaptive, threshold, max_iter, False, False)[0]
    n = len(yv)
    out = []
    for col in beta:
        m = ssum(col) / n
        out.append(ssum((v - m) ** 2 for v in col) / (n - 1))
    return out


def mgwrtst(
    y, X, coords, nsim=9, bandwidths=None, seed=0, kernel="bisquare", adaptive=False, threshold=1e-8, max_iter=500
):
    r"""Monte Carlo test of spatial non-stationarity of each MGWR coefficient.

    The Brunsdon, Fotheringham and Charlton (1998) randomisation test applied
    to MGWR: the statistic of covariate k is the sample variance of its
    local estimates; the locations are permuted nsim times (Philox
    Fisher-Yates, stream s of seed) with the bandwidths held at their
    observed-data values (selected by :func:`morie.fn.mgwrbw.mgwrbw` when
    None), and the p-value is 1 - r / (nsim + 1) with r the rank of
    the observed variance.

    References
    ----------
    Brunsdon, C., Fotheringham, A. S. and Charlton, M. (1998). Geographically
    weighted regression - modelling spatial non-stationarity. *The
    Statistician* 47, 431-443.
    Fotheringham, A. S., Yang, W. and Kang, W. (2017). Multiscale
    geographically weighted regression (MGWR). *Annals of the American
    Association of Geographers* 107, 1247-1265.

    Examples
    --------
    >>> import math
    >>> P = [(float(i % 5), float(i // 5)) for i in range(20)]
    >>> X = [[math.sin(i), (0.3 * i) % 1.1] for i in range(20)]
    >>> y = [1.0 + (1 + 0.2 * P[i][0]) * X[i][0] - X[i][1] + 0.1 * math.cos(3 * i) for i in range(20)]
    >>> r = mgwrtst(y, X, P, nsim=9, bandwidths=[6.0, 3.0, 8.0], kernel="gaussian")
    >>> [round(v, 10) for v in r["observed_variance"]]
    [1.234e-07, 0.0020085195, 6.479e-07]
    """
    yv, Xm, D = _prep(y, X, coords)
    n = len(yv)
    if bandwidths is None:
        bandwidths = _backfit(yv, Xm, D, None, kernel, adaptive, threshold, max_iter, True, False)[4]
    obs = _var_betas(yv, Xm, D, bandwidths, kernel, adaptive, threshold, max_iter)
    sims = []
    for s in range(int(nsim)):
        u = random_uniform(n, seed=seed, stream=s)
        perm = list(range(n))
        for i in range(n - 1, 0, -1):
            j = int(math.floor(float(u[i]) * (i + 1)))
            perm[i], perm[j] = perm[j], perm[i]
        Dp = [[D[perm[i]][perm[j]] for j in range(n)] for i in range(n)]
        sims.append(_var_betas(yv, Xm, Dp, bandwidths, kernel, adaptive, threshold, max_iter))
    p_values = [1.0 - (1 + sum(1 for s in sims if s[k] < obs[k])) / (nsim + 1.0) for k in range(len(obs))]
    return RichResult(
        payload={"observed_variance": obs, "simulated_variance": sims, "p_values": p_values, "bandwidths": bandwidths}
    )


mgwrtst_fn = mgwrtst


def cheatsheet() -> str:
    return "mgwrtst(y, X, coords, nsim=9) -> Monte Carlo test of MGWR coefficient variability."
