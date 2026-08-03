# morie.fn -- function file (rootcoder007/morie)
"""DPM marginal likelihood via the urn.

Implements eq. (4.13) applied per (5.2); Prop 5.2 context of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dpm_marg"]


def _norm_pdf(x, mu, sd):
    z = (x - mu) / sd
    return math.exp(-0.5 * z * z) / (sd * math.sqrt(2.0 * math.pi))


def ghosal_dpm_marg(data, alpha=1.0, tau=1.0, sigma=0.5):
    """p(X_1..X_n) = prod_i p(X_i | X_1..X_{i-1}) with the predictive
    for theta_i given by the Polya urn (4.13); normal kernel
    N(theta, sigma^2), base N(0, tau^2). For the collapsed sequential
    approximation each X_i is scored against a fresh-draw component
    (marginal N(0, tau^2 + sigma^2)) with weight M/(M+i-1) and each
    earlier latent set to its datum with weight 1/(M+i-1).
    Keys: estimate."""
    xs = _bnp._flat(data)
    M = float(alpha)
    s_marg = math.sqrt(tau * tau + sigma * sigma)
    logp = 0.0
    for i, xi in enumerate(xs):
        fresh = M / (M + i) * _norm_pdf(xi, 0.0, s_marg)
        old = sum(_norm_pdf(xi, xj, sigma) for xj in xs[:i]) \
            / (M + i)
        logp += math.log(fresh + old)
    res = RichResult(payload={"estimate": logp,
                              "n": len(xs),
                              "method": "sequential urn marginal (GvdV 2017 eq. 4.13/5.2)"})
    return with_describe_pointer(res, "gh_c5_2")


def cheatsheet():
    return "gh_c5_2: DPM marginal likelihood via the urn"
