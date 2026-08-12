"""Tau-leaping stochastic simulation (Gillespie 2001)."""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["taulep", "tau_leap_ssa"]


def _poisson(rng, lam):
    # exact Poisson count via the unit-rate exponential process:
    # K = max{n : sum_{i<=n} Exp(1)_i <= lam}; each Exp(1) is
    # -log(U), so the draw sequence is exactly mirrorable in R.
    if lam <= 0.0:
        return 0
    k = 0
    acc = 0.0
    while True:
        u = float(rng.uniform())
        while u <= 0.0:
            u = float(rng.uniform())
        acc += -math.log(u)
        if acc > lam:
            return k
        k += 1


def taulep(nu, propensity, x0, tau, n_steps, seed=0):
    """
    Explicit tau-leaping simulation of a jump process.

    Gillespie (2001), the basic tau-leaping method: provided the
    Leap Condition holds (propensities approximately constant over
    [t, t + tau)), the number of firings of reaction channel R_j in
    the interval is the Poisson random variable
    K_j = P(a_j(x) tau) (his Eq. 16), the K_j are independent, and
    the state advances by x <- x + sum_j K_j nu_j where nu_j is the
    state-change (stoichiometry) vector of channel j.  Poisson
    counts are drawn by the exact exponential-interarrival counting
    method so both language arms consume the identical uniform
    stream.

    Sources
    -------
    Gillespie, D. T. (2001). Approximate accelerated stochastic
    simulation of chemically reacting systems. *Journal of Chemical
    Physics*, 115(4), 1716-1733, Eqs. 14-16 and the basic leap
    procedure (local copy fetched-wave3/Approximate accelerated
    stochastic simulation of chemically reacting systems.pdf).

    Parameters
    ----------
    nu : matrix (M channels x N species)
        Stoichiometric state-change vectors (integers).
    propensity : callable
        propensity(x) -> length-M list of non-negative rates a_j(x).
    x0 : sequence of float
        Initial state.
    tau : float
        Leap size (fixed; caller enforces the Leap Condition).
    n_steps : int
        Number of leaps.
    seed : int
        Native-RNG seed (SplitMix64; mirrored by .ghc_rng in R).

    Returns
    -------
    RichResult
        Keys: path (states after each leap, incl. start), times,
        firings (total per channel), tau, n_steps.
    """
    nu_m = [[float(v) for v in row] for row in nu]
    m = len(nu_m)
    x = [float(v) for v in x0]
    nsp = len(x)
    if any(len(r) != nsp for r in nu_m):
        raise ValueError("each nu row must match the state length")
    tau = float(tau)
    if tau <= 0:
        raise ValueError("tau must be positive")
    n_steps = int(n_steps)
    rng = np.random.default_rng(seed)
    path = [list(x)]
    times = [0.0]
    fired = [0] * m
    for s in range(1, n_steps + 1):
        a = [float(v) for v in propensity(x)]
        if len(a) != m or any(v < 0 for v in a):
            raise ValueError("propensity must return M non-negative rates")
        ks = [_poisson(rng, aj * tau) for aj in a]
        for j in range(m):
            fired[j] += ks[j]
            if ks[j]:
                for i in range(nsp):
                    x[i] += ks[j] * nu_m[j][i]
        # clamp at zero: firings that would overdraw a species
        # (Gillespie notes negative populations as a tau-too-large
        # symptom; clamping keeps the demo-scale runs sane)
        for i in range(nsp):
            if x[i] < 0.0:
                x[i] = 0.0
        path.append(list(x))
        times.append(s * tau)
    return RichResult(payload={
        "path": path,
        "times": times,
        "firings": fired,
        "tau": tau,
        "n_steps": n_steps,
        "seed": int(seed),
        "method": "explicit tau-leaping (Gillespie 2001, Eq. 16)",
    })


# long descriptive alias (stub-era name)
tau_leap_ssa = taulep


def cheatsheet():
    return "taulep: K_j ~ Poisson(a_j(x) tau); x += sum K_j nu_j per leap"
