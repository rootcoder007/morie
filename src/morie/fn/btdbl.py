"""Double-bootstrap prepivoted confidence interval (Beran 1987)."""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["btdbl", "double_bootstrap_ci"]


def btdbl(x, statistic=None, alpha=0.05, B_outer=400, B_inner=200,
          seed=0):
    """
    Prepivoted (double) bootstrap confidence interval.

    Beran (1987): with root R_n(X, theta) = sqrt(n)|t(X) - theta|,
    the bootstrap CDF H_n estimates the root's law, and PREPIVOTING
    replaces the root by H_n(R_n) (his Sec. 2).  The confidence set
    based on the prepivoted root is computed by his Eq. 2.7 two-step
    algorithm: (a) for each outer bootstrap sample X*_b, an inner
    bootstrap gives the CDF H*_b, and the values
    u_b = H*_b(R_n(X*_b, t(X))) form the estimated law H_{n,1} of
    the prepivoted root; take its (1 - alpha) quantile c_{n,1};
    (b) the critical value is the c_{n,1} quantile of H_n itself,
    and B_{n,1} = {theta : R_n(X, theta) <= H_n^{-1}(c_{n,1})}
    (Eqs. 2.6-2.7).  Beran proves the level error is reduced by an
    order; his Sec. 3 example shows that for the mean this
    construction reproduces the classical t interval exactly in the
    limit (his Eq. 3.2).

    Sources
    -------
    Beran, R. (1987). Prepivoting to reduce level error of
    confidence sets. *Biometrika*, 74(3), 457-468, Eqs. 2.6-2.7,
    2.11-2.12 (Monte Carlo algorithm) and Eq. 3.2 (local copy
    fetched-wave3/Prepivoting_to_reduce_level_error_of_confidence_
    sets..pdf).

    Parameters
    ----------
    x : sequence of float
        Sample.
    statistic : callable, optional
        t(x) -> float (default the mean).
    alpha : float
        1 - confidence level.
    B_outer, B_inner : int
        Outer and inner bootstrap sizes.
    seed : int
        Native-RNG seed (SplitMix64; mirrored by .ghc_rng in R).

    Returns
    -------
    RichResult
        Keys: estimate, lower, upper, critical_root, c_level
        (c_{n,1}), alpha.
    """
    xv = [float(v) for v in x]
    n = len(xv)
    if n < 5:
        raise ValueError("need at least five observations")
    if statistic is None:
        statistic = lambda s: sum(s) / len(s)
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    rng = np.random.default_rng(seed)
    that = float(statistic(xv))
    sqn = math.sqrt(n)

    def _resample(base):
        return [base[min(int(float(rng.uniform()) * n), n - 1)]
                for _ in range(n)]

    outer_roots = []
    prepiv = []
    for _b in range(int(B_outer)):
        xb = _resample(xv)
        tb = float(statistic(xb))
        rb = sqn * abs(tb - that)
        outer_roots.append(rb)
        # inner bootstrap CDF at rb
        le = 0
        for _c in range(int(B_inner)):
            xc = _resample(xb)
            tc = float(statistic(xc))
            if sqn * abs(tc - tb) <= rb:
                le += 1
        prepiv.append(le / float(B_inner))
    # step (a): c_{n,1} = (1 - alpha) quantile of the prepivoted values
    sp = sorted(prepiv)
    idx = min(int(math.ceil((1.0 - alpha) * B_outer)) - 1, B_outer - 1)
    c1 = sp[max(idx, 0)]
    # step (b): critical root = c1 quantile of H_n
    so = sorted(outer_roots)
    j = min(int(math.ceil(c1 * B_outer)) - 1, B_outer - 1)
    crit = so[max(j, 0)]
    half = crit / sqn
    return RichResult(payload={
        "estimate": that,
        "lower": that - half,
        "upper": that + half,
        "critical_root": crit,
        "c_level": c1,
        "alpha": alpha,
        "B_outer": int(B_outer),
        "B_inner": int(B_inner),
        "seed": int(seed),
        "method": "Beran (1987) prepivoted double bootstrap (Eq. 2.7)",
    })


# long descriptive alias (stub-era name)
double_bootstrap_ci = btdbl


def cheatsheet():
    return "btdbl: c1 = q_{1-a}(H*_b(R*_b)); crit = H_n^{-1}(c1); theta in t +- crit/sqrt(n)"
