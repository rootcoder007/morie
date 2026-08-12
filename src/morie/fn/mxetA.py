"""Max-stable process simulation (de Haan 1984 spectral representation)."""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["mxetA", "max_stable_simulation"]


def mxetA(F, n_sim=1, seed=0, max_points=100000):
    """
    Simulate a max-stable process by de Haan's spectral construction.

    de Haan (1984), the constructive example of Sec. 1 (p. 1195):
    take a Poisson point process on R_+ x [0, 1] with intensity
    (dx/x^2) dt and points {(X_k, T_k)}; for non-negative spectral
    functions f_t the field

        Y_t = sup_k f_t(T_k) X_k

    is max-stable with P(Y_t <= y) = exp(-(1/y) int_0^1 f_t(s) ds)
    (his displayed calculation).  The intensity dx/x^2 makes
    X_k = 1/Gamma_k with Gamma_k the arrival times of a unit-rate
    Poisson process, and T_k iid uniform.  Simulation stops exactly
    when 1/Gamma_k max_{t,s} f_t(s) falls below the current minimum
    of Y over t: no later point can alter any coordinate, so the
    truncation is exact rather than approximate.  Spectral functions
    are supplied discretized on m uniform sites of [0, 1], so the
    marginal scale is c_t = mean_s F[t, s].

    Sources
    -------
    de Haan, L. (1984). A spectral representation for max-stable
    processes. *Annals of Probability*, 12(4), 1194-1204, Sec. 1
    example and Theorem 3 (local copy fetched-wave3/A spectral
    representation for max-stable processes.pdf; formulas read from
    the rendered scan -- no text layer).

    Parameters
    ----------
    F : matrix (n_t x m)
        Non-negative spectral functions f_t sampled on m uniform
        sites of [0, 1].
    n_sim : int
        Number of independent field realizations.
    seed : int
        Native-RNG seed (SplitMix64; mirrored by .ghc_rng in R).
    max_points : int
        Safety cap on Poisson points per realization.

    Returns
    -------
    RichResult
        Keys: fields (n_sim x n_t), scales (c_t), n_points (per
        realization), frechet_uniform (exp(-c_t / Y_t) values,
        each exactly U(0,1) in law).
    """
    Fm = [[float(v) for v in row] for row in F]
    nt = len(Fm)
    m = len(Fm[0])
    if any(len(r) != m for r in Fm) or nt < 1 or m < 1:
        raise ValueError("F must be a rectangular n_t x m matrix")
    if any(v < 0 for row in Fm for v in row):
        raise ValueError("spectral functions must be non-negative")
    fmax = max(v for row in Fm for v in row)
    if fmax <= 0:
        raise ValueError("F must not be identically zero")
    scales = [sum(row) / m for row in Fm]
    rng = np.random.default_rng(seed)
    fields = []
    counts = []
    for _ in range(int(n_sim)):
        y = [0.0] * nt
        gamma = 0.0
        k = 0
        while k < max_points:
            u = float(rng.uniform())
            while u <= 0.0:
                u = float(rng.uniform())
            gamma += -math.log(u)
            x = 1.0 / gamma
            if x * fmax <= min(y) and min(y) > 0.0:
                break                      # exact truncation
            site = min(int(float(rng.uniform()) * m), m - 1)
            for t in range(nt):
                v = Fm[t][site] * x
                if v > y[t]:
                    y[t] = v
            k += 1
        fields.append(y)
        counts.append(k)
    fu = [[math.exp(-scales[t] / y[t]) if y[t] > 0 else 0.0
           for t in range(nt)] for y in fields]
    return RichResult(payload={
        "fields": fields,
        "scales": scales,
        "n_points": counts,
        "frechet_uniform": fu,
        "seed": int(seed),
        "method": "de Haan (1984) spectral max-stable simulation",
    })


# long descriptive alias (stub-era name)
max_stable_simulation = mxetA


def cheatsheet():
    return "mxetA: Y_t = sup_k f_t(T_k)/Gamma_k; P(Y<=y) = exp(-c_t/y)"
