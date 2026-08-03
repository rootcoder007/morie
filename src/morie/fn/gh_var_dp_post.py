# morie.fn -- function file (rootcoder007/morie)
"""Variational DP posterior.

Implements sec. 5.3 (truncated mean-field family) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_variational_dp_posterior"]


def ghosal_variational_dp_posterior(x=None, K=8, alpha=1.0, tau=1.0,
                                    sigma=0.5, n_iter=60, seed=42):
    """q*(G_K, theta, z) = argmin_q KL(q || pi(. | X)) over the
    truncated mean-field family (sec. 5.3): coordinate ascent on the
    truncated stick-breaking model (Example 5.5 updates). Returns the
    fitted component centers and responsibilities entropy.
    Keys: posterior."""
    if x is None:
        x = [-2.1, -1.9, -2.0, 1.9, 2.0, 2.1]
    xs = _bnp._flat(x)
    n = len(xs)
    lo, hi = min(xs), max(xs)
    span = (hi - lo) or 1.0
    xi = [lo + span * (j + 0.5) / K for j in range(int(K))]
    r = [[1.0 / K] * int(K) for _ in range(n)]
    for _ in range(int(n_iter)):
        for i in range(n):
            logits = [-(xs[i] - xi[j]) ** 2
                      / (2.0 * sigma * sigma) for j in range(int(K))]
            mx = max(logits)
            ex = [math.exp(v - mx) for v in logits]
            tot = sum(ex)
            r[i] = [v / tot for v in ex]
        for j in range(int(K)):
            num = sum(r[i][j] * xs[i] for i in range(n)) \
                / sigma ** 2
            den = sum(r[i][j] for i in range(n)) / sigma ** 2 \
                + 1.0 / tau ** 2
            xi[j] = num / den
    ent = -sum(r[i][j] * math.log(max(r[i][j], 1e-300))
               for i in range(n) for j in range(int(K))) / n
    res = RichResult(payload={"estimate": ent, "posterior": xi,
                              "centers": xi,
                              "method": "variational DP posterior (GvdV 2017 sec. 5.3)"})
    return with_describe_pointer(res, "gh_var_dp_post")


def cheatsheet():
    return "gh_var_dp_post: Variational DP posterior"
