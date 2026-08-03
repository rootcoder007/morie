# morie.fn -- function file (rootcoder007/morie)
"""Universal model weights.

Implements sec. 10.2.1 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_univ_weights"]


def ghosal_univ_weights(n=100, c=2.0, K_max=200, eps_scale=1.0):
    """Weights pi_k propto exp(-c k log n) make
    sum_k pi_k exp(n eps_k^2) converge whenever n eps_k^2 <~ k log n
    (sec. 10.2.1) -- the universal-weight condition behind
    adaptation. Computes the partial sums. Keys: estimate."""
    log_pis = [-c * k * math.log(n) for k in range(1, K_max + 1)]
    mx = max(log_pis)
    Z = sum(math.exp(v - mx) for v in log_pis)
    total = 0.0
    partial = []
    for k in range(1, K_max + 1):
        n_eps2 = eps_scale * k * math.log(n)     # eps_k^2 = k log n/n
        total += math.exp(log_pis[k - 1] - mx - math.log(Z)
                          + n_eps2)
        if k in (10, 50, K_max):
            partial.append(total)
    converged = partial[-1] < 10.0 * partial[0] + 1e9 and \
        math.isfinite(total)
    res = RichResult(payload={"estimate": total,
                              "partial_sums": partial,
                              "converges": math.isfinite(total)
                              and c > eps_scale,
                              "method": "universal weights (GvdV 2017 sec. 10.2.1)"})
    return with_describe_pointer(res, "gh_c10_2")


def cheatsheet():
    return "gh_c10_2: Universal model weights"
