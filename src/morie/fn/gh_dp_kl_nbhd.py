# morie.fn -- function file (rootcoder007/morie)
"""Prior mass of KL neighborhoods.

Implements sec. 7.2 (KL-neighborhood lower bounds); Lemma 6.26 form of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dp_kl_nbhd_mass"]


def ghosal_dp_kl_nbhd_mass(p0, alpha=None, eps_list=(0.4, 0.2, 0.1),
                           n_sim=4000, seed=42):
    """Rates come from lower bounds on Pi(K(p0; p) < eps)
    (sec. 7.2; the evidence bound of Lemma 6.26 turns this mass into
    a denominator bound). Monte Carlo log-mass under a Dirichlet
    prior for shrinking eps: positive for every eps = KL property,
    with log-mass decreasing as eps shrinks. Keys: estimate."""
    p0 = _bnp.normalize_weights(p0)
    k = len(p0)
    if alpha is None:
        alpha = [1.0] * k
    rng = np.random.default_rng(seed)
    kls = []
    for _ in range(n_sim):
        g = [float(rng.gamma(a, 1.0)) for a in alpha]
        p = _bnp.normalize_weights(g)
        kls.append(sum(q * math.log(q / max(pi, 1e-300))
                       for q, pi in zip(p0, p) if q > 0))
    log_masses = []
    for e in eps_list:
        hits = sum(1 for v in kls if v < e)
        log_masses.append(math.log(max(hits, 1) / n_sim))
    res = RichResult(payload={"estimate": log_masses[-1],
                              "log_mass_by_eps": log_masses,
                              "monotone": all(
                                  log_masses[i + 1] <= log_masses[i]
                                  + 1e-12
                                  for i in range(len(log_masses)
                                                 - 1)),
                              "method": "KL-neighborhood prior mass (GvdV 2017 sec. 7.2)"})
    return with_describe_pointer(res, "gh_dp_kl_nbhd")


def cheatsheet():
    return "gh_dp_kl_nbhd: Prior mass of KL neighborhoods"
