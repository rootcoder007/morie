# morie.fn -- function file (rootcoder007/morie)
"""GWR Monte-Carlo test for spatial variability."""

import math

from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_uniform
from .gwrbas import gwr_kernel_weights
from .gwrcoef import _flat, _rows, _with_intercept, _wls


def _betas_var(yv, Xm, P, bw, kernel, adaptive):
    n, p = len(yv), len(Xm[0])
    B = []
    for i in range(n):
        d = [math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]) for j in range(n)]
        B.append(_wls(Xm, gwr_kernel_weights(d, bw, kernel, adaptive), yv)[0])
    out = []
    for a in range(p):
        col = [r[a] for r in B]
        m = ssum(col) / n
        out.append(ssum((v - m) ** 2 for v in col) / (n - 1))
    return out


def gwrtst(y, X, coords, bw=0.5, nsim=9, seed=0, kernel="bisquare", adaptive=False):
    r"""Monte Carlo test of spatial variability of each GWR coefficient (Brunsdon, Fotheringham and Charlton 1998).

    The statistic for coefficient k is the sample variance (n - 1
    divisor) of its local estimates. Under the null of a stationary
    coefficient the locations are exchangeable, so the coordinates are
    randomly permuted (Philox-driven Fisher-Yates, stream s of seed
    for simulation s), the GWR refitted, and the p-value is 1 -
    r / (nsim + 1) with r the rank of the observed variance among the
    nsim + 1 values, as GWmodel::gwr.montecarlo.

    References
    ----------
    Brunsdon, C., Fotheringham, A. S. and Charlton, M. (1998). Geographically
    weighted regression - modelling spatial non-stationarity. *The
    Statistician* 47, 431-443.

    Examples
    --------
    >>> P = [(float(i % 4), float(i // 4)) for i in range(16)]
    >>> X = [[(0.3 * i) % 1.7] for i in range(16)]
    >>> y = [1.0 + 2.0 * X[i][0] + 0.1 * P[i][0] + 0.05 * (i % 3) for i in range(16)]
    >>> r = gwrtst(y, X, P, 3.0, nsim=19, kernel="gaussian")
    >>> [round(v, 10) for v in r["observed_variance"]]
    [0.0002535631, 5.07873e-05]
    """
    yv = _flat(y)
    n = len(yv)
    Xm = _with_intercept(X, n)
    P = _rows(coords)
    obs = _betas_var(yv, Xm, P, bw, kernel, adaptive)
    sims = []
    for s in range(int(nsim)):
        u = random_uniform(n, seed=seed, stream=s)
        perm = list(range(n))
        for i in range(n - 1, 0, -1):
            j = int(math.floor(float(u[i]) * (i + 1)))
            perm[i], perm[j] = perm[j], perm[i]
        sims.append(_betas_var(yv, Xm, [P[perm[i]] for i in range(n)], bw, kernel, adaptive))
    p_values = []
    for a in range(len(obs)):
        rank = 1 + sum(1 for s in sims if s[a] < obs[a])
        p_values.append(1.0 - rank / (nsim + 1.0))
    return RichResult(payload={"observed_variance": obs, "simulated_variance": sims, "p_values": p_values})


gwrtst_fn = gwrtst


def cheatsheet() -> str:
    return "gwrtst(y, X, coords, bw, nsim=9) -> Monte Carlo test of coefficient variability (GWmodel::gwr.montecarlo)."
