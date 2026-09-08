"""Morris elementary effects screening (Morris 1991; Campolongo 2007)."""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["morrisM", "morris_screening"]


def morrisM(fun, k, r=10, p=4, seed=0, bounds=None):
    """
    Elementary effects screening of input factors.

    Morris (1991) as formalized in Saltelli et al. (2008), Sec. 3.2,
    Eq. 3.1: on a p-level grid of the unit cube, the elementary
    effect of factor i at point X is

        EE_i = [ Y(X + Delta e_i) - Y(X) ] / Delta,

    with Delta = p / (2 (p - 1)) (the recommended choice for even p).
    Each of the r trajectories draws a random grid base point and
    perturbs the factors one at a time in random order with random
    direction (+Delta or -Delta, flipped when the step would leave
    the cube), costing r (k + 1) model runs.  Reported per factor:
    mu (mean of EE_i, Morris), sigma (standard deviation of EE_i,
    Morris), and mu_star (mean of |EE_i|, Campolongo et al. 2007,
    which avoids the type-II cancellation of mu).

    Sources
    -------
    Morris, M. D. (1991). Factorial sampling plans for preliminary
    computational experiments. *Technometrics*, 33, 161-174.
    Campolongo, F., Cariboni, J. & Saltelli, A. (2007). An effective
    screening design for sensitivity analysis of large models.
    *Environmental Modelling & Software*, 22, 1509-1518.
    Saltelli, A. et al. (2008). *Global Sensitivity Analysis: The
    Primer*, Wiley, Sec. 3.2-3.3 (local copy
    fetched-wave3/saltelli-2008-gsa-primer.pdf).

    Parameters
    ----------
    fun : callable
        Model Y = fun(x) taking a length-k list (in original units).
    k : int
        Number of input factors.
    r : int
        Number of trajectories.
    p : int
        Number of grid levels (even; Delta = p/(2(p-1))).
    seed : int
        Native-RNG seed (SplitMix64; mirrored by .ghc_rng in R).
    bounds : list of (low, high), optional
        Per-factor ranges (default unit cube).

    Returns
    -------
    RichResult
        Keys: mu, mu_star, sigma (length-k lists), n_runs, delta,
        ee (per-factor lists of elementary effects).
    """
    k = int(k)
    r = int(r)
    p = int(p)
    if k < 1 or r < 1:
        raise ValueError("k and r must be positive")
    if p < 2 or p % 2:
        raise ValueError("p must be an even integer >= 2")
    if bounds is None:
        bounds = [(0.0, 1.0)] * k
    bounds = [(float(lo), float(hi)) for lo, hi in bounds]
    if len(bounds) != k or any(hi <= lo for lo, hi in bounds):
        raise ValueError("bounds must be k pairs with low < high")
    delta = p / (2.0 * (p - 1.0))
    # grid levels for base points: {0, 1/(p-1), ..., 1 - delta}
    levels = [j / (p - 1.0) for j in range(p) if j / (p - 1.0) <= 1.0 - delta + 1e-12]
    rng = np.random.default_rng(seed)
    ee = [[] for _ in range(k)]
    n_runs = 0

    def _scale(u):
        return [bounds[i][0] + u[i] * (bounds[i][1] - bounds[i][0])
                for i in range(k)]

    for _ in range(r):
        # all randomness from sequential scalar uniforms so the R arm
        # (.ghc_unif) can mirror the stream draw for draw
        x = [levels[min(int(float(rng.uniform()) * len(levels)),
                        len(levels) - 1)] for _ in range(k)]
        keys = [float(rng.uniform()) for _ in range(k)]
        order = sorted(range(k), key=lambda i: keys[i])
        y = float(fun(_scale(x)))
        n_runs += 1
        for i in order:
            step = delta if float(rng.uniform()) < 0.5 else -delta
            if x[i] + step > 1.0 + 1e-12 or x[i] + step < -1e-12:
                step = -step
            x2 = list(x)
            x2[i] = x[i] + step
            y2 = float(fun(_scale(x2)))
            n_runs += 1
            # EE in the scaled (unit-cube) parameterization, Eq. 3.1
            ee[i].append((y2 - y) / step)
            x = x2
            y = y2
    mu = [sum(v) / len(v) for v in ee]
    mu_star = [sum(abs(x_) for x_ in v) / len(v) for v in ee]
    sigma = []
    for v, m in zip(ee, mu):
        if len(v) > 1:
            sigma.append(math.sqrt(sum((x_ - m) ** 2 for x_ in v)
                                   / (len(v) - 1)))
        else:
            sigma.append(float("nan"))
    return RichResult(payload={
        "mu": mu, "mu_star": mu_star, "sigma": sigma,
        "ee": ee, "n_runs": n_runs, "delta": delta,
        "r": r, "p": p, "seed": int(seed),
        "method": "Morris elementary effects (Saltelli 2008 Eq. 3.1)",
    })


# long descriptive alias (stub-era name)
morris_screening = morrisM


def cheatsheet():
    return "morrisM: EE_i = [Y(X+De_i)-Y(X)]/D; report mu, mu*, sigma"
