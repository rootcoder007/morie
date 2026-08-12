"""Iterated prepivoted bootstrap test (Beran 1988)."""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["btiseq", "iterated_bootstrap_test"]


def btiseq(x, mu0=0.0, B_outer=300, B_inner=150, seed=0):
    """
    Prepivoted and twice-prepivoted bootstrap tests for a mean.

    Beran (1988): prepivoting transforms a test statistic by the cdf
    of its bootstrap null distribution; the simple bootstrap test
    phi_B refers the prepivoted statistic H_n(T_n) to a Uniform(0,1)
    quantile -- equivalently it rejects on the bootstrap p-value
    p_B = 1 - H_n(T_n).  The prepivoted bootstrap test phi_B1
    prepivots TWICE: each outer null resample X*_b yields its own
    inner bootstrap cdf H*_b, and the second-level p-value is the
    fraction of outer resamples whose prepivoted statistic
    H*_b(T(X*_b)) exceeds H_n(T_n); under regularity the level
    error of phi_B1 is of smaller asymptotic order than phi_B or the
    asymptotic test (his abstract and Sec. 2).  Null resampling for
    the mean test draws from the recentered sample x - xbar + mu0.

    Sources
    -------
    Beran, R. (1988). Prepivoting test statistics: a bootstrap view
    of asymptotic refinements. *JASA*, 83(403), 687-697, Secs. 1-2
    (local copy fetched-wave3/Prepivoting_Test_Statistics_A_
    Bootstrap_View_of_Asymptotic_Refinements.pdf).

    Parameters
    ----------
    x : sequence of float
        Sample.
    mu0 : float
        Null value of the mean.
    B_outer, B_inner : int
        Outer and inner bootstrap sizes.
    seed : int
        Native-RNG seed (SplitMix64; mirrored by .ghc_rng in R).

    Returns
    -------
    RichResult
        Keys: statistic (studentized |T|), p_boot (single
        prepivot), p_iterated (double prepivot), prepivoted_value
        (H_n(T_n)).
    """
    xv = [float(v) for v in x]
    n = len(xv)
    if n < 5:
        raise ValueError("need at least five observations")
    mu0 = float(mu0)
    rng = np.random.default_rng(seed)

    def _tstat(s, center):
        m = sum(s) / n
        sd = math.sqrt(sum((v - m) ** 2 for v in s) / (n - 1))
        if sd <= 0:
            return 0.0
        return math.sqrt(n) * abs(m - center) / sd

    def _resample(base):
        return [base[min(int(float(rng.uniform()) * n), n - 1)]
                for _ in range(n)]

    t_obs = _tstat(xv, mu0)
    xbar = sum(xv) / n
    null_x = [v - xbar + mu0 for v in xv]
    outer_t = []
    prepiv_outer = []
    for _b in range(int(B_outer)):
        xb = _resample(null_x)
        tb = _tstat(xb, mu0)
        outer_t.append(tb)
        # inner prepivot of tb under xb's own null resampling
        mb = sum(xb) / n
        null_b = [v - mb + mu0 for v in xb]
        le = 0
        for _c in range(int(B_inner)):
            xc = _resample(null_b)
            if _tstat(xc, mu0) <= tb:
                le += 1
        prepiv_outer.append(le / float(B_inner))
    h_obs = sum(1 for t in outer_t if t <= t_obs) / float(B_outer)
    p_boot = 1.0 - h_obs
    p_iter = sum(1 for u in prepiv_outer if u >= h_obs) / float(B_outer)
    return RichResult(payload={
        "statistic": t_obs,
        "p_boot": p_boot,
        "p_iterated": p_iter,
        "prepivoted_value": h_obs,
        "mu0": mu0,
        "B_outer": int(B_outer),
        "B_inner": int(B_inner),
        "seed": int(seed),
        "method": "Beran (1988) prepivoted / twice-prepivoted test",
    })


# long descriptive alias (stub-era name)
iterated_bootstrap_test = btiseq


def cheatsheet():
    return "btiseq: p_B = 1 - H(T); p_B1 = frac{H*_b(T*_b) >= H(T)}"
