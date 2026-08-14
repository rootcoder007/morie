"""Rao-Wu rescaled survey bootstrap (Rao, Wu & Yue 1992)."""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["bootss", "rao_wu_bootstrap"]


def bootss(y, weights, strata, clusters, statistic=None, B=200,
           m=None, seed=0):
    """
    Rescaled bootstrap for stratified multistage survey data.

    Rao, Wu & Yue (1992), Sec. 3.3: within each stratum h draw a
    simple random sample of m_h clusters WITH replacement from the
    n_h sample clusters; with m*_hi the number of times cluster (hi)
    is selected, the bootstrap weights are (their Eq. 3.4)

        w*_hik = [ {1 - sqrt(m_h/(n_h-1))}
                   + sqrt(m_h/(n_h-1)) (n_h/m_h) m*_hi ] w_hik,

    and the bootstrap variance estimator is (Eq. 3.5)

        s^2_BOOT = (1/B) sum_b [theta*_(b) - theta_hat]^2.

    With the standard choice m_h = n_h - 1 the weights are always
    non-negative, and for n_h = 2 a selected cluster's weight doubles
    while the other drops to zero (as printed in the paper).  In the
    linear case s^2_BOOT reduces to the customary with-replacement
    variance estimator s^2(Y_hat) of their Eq. 2.3.

    Sources
    -------
    Rao, J. N. K., Wu, C. F. J. & Yue, K. (1992). Some recent work
    on resampling methods for complex surveys. *Survey Methodology*,
    18(2), 209-217, Eqs. 2.1-2.3, 3.4, 3.5 (local copy
    fetched-wave3/rao-wu-yue-1992-survey-bootstrap.pdf).
    Rao, J. N. K. & Wu, C. F. J. (1988). Resampling inference with
    complex survey data. *JASA*, 83, 231-241 (the rescaling idea).

    Parameters
    ----------
    y : sequence of float
        Observations (ultimate units).
    weights : sequence of float
        Survey weights w_hik (> 0).
    strata : sequence
        Stratum label per observation.
    clusters : sequence
        Cluster (PSU) label per observation, unique within stratum.
    statistic : callable, optional
        statistic(y, w) -> float; default the weighted total
        sum(w_i y_i) (their Eq. 2.1).
    B : int
        Number of bootstrap replicates.
    m : int or dict, optional
        Clusters drawn per stratum; default n_h - 1 per stratum.
    seed : int
        Native-RNG seed (SplitMix64; mirrored by .ghc_rng in R).

    Returns
    -------
    RichResult
        Keys: estimate (theta_hat), variance (Eq. 3.5), se,
        replicates, B, n_strata.
    """
    yv = [float(v) for v in y]
    wv = [float(v) for v in weights]
    n = len(yv)
    if not (len(wv) == len(strata) == len(clusters) == n) or n < 2:
        raise ValueError("y, weights, strata, clusters must be paired")
    if any(v <= 0 for v in wv):
        raise ValueError("weights must be positive")
    if statistic is None:
        statistic = lambda yy, ww: sum(a * b for a, b in zip(ww, yy))
    # index clusters within strata (insertion order = data order)
    strat_order = []
    clus = {}
    for i in range(n):
        h = strata[i]
        if h not in clus:
            clus[h] = {}
            strat_order.append(h)
        c = clusters[i]
        clus[h].setdefault(c, []).append(i)
    for h in strat_order:
        if len(clus[h]) < 2:
            raise ValueError("every stratum needs >= 2 clusters")
    mh = {}
    for h in strat_order:
        nh = len(clus[h])
        if m is None:
            mh[h] = nh - 1
        elif isinstance(m, dict):
            mh[h] = int(m[h])
        else:
            mh[h] = int(m)
        if not (1 <= mh[h] <= nh - 1):
            raise ValueError("need 1 <= m_h <= n_h - 1 for "
                             "non-negative weights")
    theta = float(statistic(yv, wv))
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(int(B)):
        wb = list(wv)
        for h in strat_order:
            cl_names = list(clus[h].keys())
            nh = len(cl_names)
            m_h = mh[h]
            counts = [0] * nh
            for _d in range(m_h):
                counts[min(int(float(rng.uniform()) * nh), nh - 1)] += 1
            root = math.sqrt(m_h / (nh - 1.0))
            for ci, cname in enumerate(cl_names):
                factor = (1.0 - root) + root * (nh / m_h) * counts[ci]
                for idx in clus[h][cname]:
                    wb[idx] = wv[idx] * factor
        reps.append(float(statistic(yv, wb)))
    var = sum((t - theta) ** 2 for t in reps) / len(reps)
    return RichResult(payload={
        "estimate": theta,
        "variance": var,
        "se": math.sqrt(var),
        "replicates": reps,
        "B": int(B),
        "n_strata": len(strat_order),
        "seed": int(seed),
        "method": "Rao-Wu-Yue rescaled bootstrap (Eqs. 3.4-3.5)",
    })


# long descriptive alias (stub-era name)
rao_wu_bootstrap = bootss


def cheatsheet():
    return "bootss: w* = [(1-r) + r (n/m) m*] w, r = sqrt(m/(n-1)); var Eq. 3.5"

# public names resolved by fn/_lazy_map.json
bootstrap_survey = bootss
