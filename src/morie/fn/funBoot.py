"""Curve-level functional bootstrap bands (Cuevas, Febrero & Fraiman 2006)."""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["funBoot", "functional_bootstrap_band"]


def funBoot(curves, statistic=None, alpha=0.05, B=500, metric="l2",
            smooth=0.0, seed=0):
    """
    Bootstrap tolerance band for a functional estimator.

    Cuevas, Febrero & Fraiman (2006), Sec. 3(e): draw B curve-level
    bootstrap samples from the observed curves x_1(t), ..., x_n(t);
    for each, evaluate the functional estimator T*; the "(1 - alpha)
    bootstrap confidence band" is the ball of radius D centred at
    the bootstrap average of T, where D is chosen so that
    (1 - alpha) of the bootstrap replications lie within distance D
    of their average -- with either the L2 or the sup (L-infinity)
    metric, exactly as in the paper.  Their smoothed-bootstrap
    variant adds Gaussian perturbations z_i(t) with standard
    deviation ``smooth`` to each resampled curve.

    Sources
    -------
    Cuevas, A., Febrero, M. & Fraiman, R. (2006). On the use of the
    bootstrap for estimating functions with functional data.
    *Computational Statistics & Data Analysis*, 51(2), 1063-1074,
    Sec. 3(e) (local copy fetched-wave3/On_the_use_of_the_bootstrap_
    for_estimating_functions_with_functional_data.pdf).

    Parameters
    ----------
    curves : matrix (n curves x m grid points)
        Discretized functional data.
    statistic : callable, optional
        T(list of curves) -> curve (default the pointwise mean).
    alpha : float
        1 - band level.
    B : int
        Bootstrap samples.
    metric : str
        "l2" (grid-averaged) or "sup".
    smooth : float
        Std dev of the smoothed-bootstrap Gaussian perturbation
        (0 = naive bootstrap).
    seed : int
        Native-RNG seed (SplitMix64; mirrored by .ghc_rng in R).

    Returns
    -------
    RichResult
        Keys: center (bootstrap average of T), radius (D),
        estimate (T on the original sample), distances,
        n_within (count inside D, = ceil((1-alpha)B)).
    """
    X = [[float(v) for v in row] for row in curves]
    n = len(X)
    m = len(X[0])
    if n < 3 or any(len(r) != m for r in X):
        raise ValueError("curves must be rectangular with n >= 3")
    if statistic is None:
        statistic = lambda cs: [sum(c[j] for c in cs) / len(cs)
                                for j in range(m)]
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    met = str(metric).lower()
    if met not in ("l2", "sup"):
        raise ValueError("metric must be 'l2' or 'sup'")
    smooth = float(smooth)
    rng = np.random.default_rng(seed)
    t_obs = [float(v) for v in statistic(X)]
    reps = []
    for _b in range(int(B)):
        sample = []
        for _i in range(n):
            pick = X[min(int(float(rng.uniform()) * n), n - 1)]
            if smooth > 0.0:
                sample.append([pick[j] + smooth * float(rng.normal())
                               for j in range(m)])
            else:
                sample.append(pick)
        reps.append([float(v) for v in statistic(sample)])
    center = [sum(r[j] for r in reps) / len(reps) for j in range(m)]

    def _dist(a, b):
        if met == "sup":
            return max(abs(a[j] - b[j]) for j in range(m))
        return math.sqrt(sum((a[j] - b[j]) ** 2 for j in range(m)) / m)

    dists = [_dist(r, center) for r in reps]
    sd = sorted(dists)
    idx = max(min(int(math.ceil((1.0 - alpha) * B)) - 1, B - 1), 0)
    D = sd[idx]
    n_within = sum(1 for d in dists if d <= D)
    return RichResult(payload={
        "center": center,
        "radius": D,
        "estimate": t_obs,
        "distances": dists,
        "n_within": n_within,
        "metric": met,
        "alpha": alpha,
        "B": int(B),
        "seed": int(seed),
        "method": "functional bootstrap band (Cuevas et al. 2006, Sec. 3e)",
    })


# long descriptive alias (stub-era name)
functional_bootstrap_band = funBoot


def cheatsheet():
    return "funBoot: D = q_{1-a}(dist(T*_b, mean T*)); band = ball(center, D)"
