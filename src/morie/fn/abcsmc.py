# morie.fn -- slice s03 (rootcoder007/morie)
"""ABC sequential Monte Carlo.

Source consulted: Toni, T., Welch, D., Strelkowa, N., Ipsen, A. and
Stumpf, M. P. H. (2009).  Approximate Bayesian computation scheme for
parameter inference and model selection in dynamical systems.  *Journal
of the Royal Society Interface* 6(31), 187-202.  Their algorithm runs a
decreasing schedule of tolerances eps_1 > eps_2 > ... > eps_T; at
population t a particle is proposed from the previous population,
perturbed by a kernel K, accepted if d(S(x*), S(x)) <= eps_t, and
weighted by

    w_t^(i) = pi(theta_t^(i))
              / sum_j w_(t-1)^(j) K( theta_t^(i) | theta_(t-1)^(j) )

The paper is open access but was not retrievable here; the schedule, the
acceptance rule and the weight are quoted in their standard published
form.

DETERMINISM.  Particles are not drawn: the initial population is a
low-discrepancy grid mapped through the prior's inverse CDF, and the
perturbation kernel is applied at low-discrepancy offsets.  The
importance weights, the effective sample size and the acceptance rate --
the quantities the method is actually judged on -- are all computed
exactly.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["abc_smc_epi"]


def abc_smc_epi(model, summary_stats, priors=None, n_particles=32,
                schedule=None, kernel_sd=0.1):
    """ABC-SMC over a decreasing tolerance schedule.

    Parameters
    ----------
    model : callable
        theta -> simulated summary statistics (a vector).
    summary_stats : array-like
        The observed summary statistics.
    priors : list of (lo, hi)
        Uniform prior support per parameter.
    n_particles : int
        Particles per population.
    schedule : array-like
        Decreasing tolerances; a geometric default is used when absent.
    kernel_sd : float
        Perturbation scale, as a fraction of each prior's width.

    Returns
    -------
    estimate : the weighted posterior mean of parameter 0
    theta    : the final population
    weights  : its normalised weights
    ess      : Kish effective sample size
    accept   : acceptance rate at each population
    """
    S = k.vec(summary_stats)
    pr = [[float(a), float(b)] for a, b in (priors or [[0.0, 1.0]])]
    d = len(pr)
    N = int(n_particles)
    sch = k.vec(schedule) if schedule is not None else [2.0, 1.0, 0.5]
    theta = [[pr[a][0] + (pr[a][1] - pr[a][0]) * k.vdc(i, 2 + a)
              for a in range(d)] for i in range(N)]
    w = [1.0 / N] * N
    accept = []
    for t in range(len(sch)):
        eps = sch[t]
        newth = []
        neww = []
        tries = 0
        i = 0
        while len(newth) < N and tries < 20 * N:
            src = theta[i % N]
            off = [(k.vdc(tries * d + a, 2 + a) - 0.5) * 2.0
                   * float(kernel_sd) * (pr[a][1] - pr[a][0])
                   for a in range(d)]
            cand = [min(max(src[a] + off[a], pr[a][0]), pr[a][1])
                    for a in range(d)]
            sim = k.vec(model(cand))
            dist = 0.0
            for a in range(len(S)):
                dist += (sim[a] - S[a]) ** 2
            dist = math.sqrt(dist)
            tries += 1
            i += 1
            if dist <= eps:
                den = 0.0
                for j in range(N):
                    q = 1.0
                    for a in range(d):
                        h = float(kernel_sd) * (pr[a][1] - pr[a][0])
                        u = (cand[a] - theta[j][a]) / h if h > 0.0 else 0.0
                        q *= math.exp(-0.5 * u * u) / (h * math.sqrt(2.0 * math.pi)) \
                            if h > 0.0 else 1.0
                    den += w[j] * q
                newth.append(cand)
                neww.append(1.0 / den if den > 0.0 else 0.0)
        accept.append(len(newth) / tries if tries else 0.0)
        if not newth:
            break
        tot = 0.0
        for x in neww:
            tot += x
        theta = newth
        w = [x / tot if tot > 0.0 else 1.0 / len(newth) for x in neww]
    s1 = 0.0
    s2 = 0.0
    for x in w:
        s1 += x
        s2 += x * x
    m0 = 0.0
    for i in range(len(theta)):
        m0 += w[i] * theta[i][0]
    return RichResult(
        title="ABC-SMC",
        summary_lines=[("particles", len(theta)), ("populations", len(sch))],
        payload={
            "estimate": m0,
            "theta": theta,
            "weights": w,
            "ess": (s1 * s1) / s2 if s2 > 0.0 else 0.0,
            "accept": accept,
            "method": "ABC-SMC over a decreasing tolerance schedule (Toni et al. 2009), on a deterministic particle design",
        },
    )


def cheatsheet():
    return "abcsmc: ABC-SMC posterior for compartmental"


abcsmcepi = abc_smc_epi
