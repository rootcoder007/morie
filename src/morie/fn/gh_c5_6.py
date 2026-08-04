# morie.fn -- function file (rootcoder007/morie)
"""Variational algorithm for DPM.

Implements eq. (5.8) + Example 5.5 updates (i)-(ii) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_vb_dpm"]


def ghosal_vb_dpm(data, K=8, alpha=1.0, tau=1.0, sigma=0.5,
                  n_iter=60, seed=42):
    """Mean-field VB: minimize KL(q; pi(.|X)) (eq. 5.8) over
    q = prod pi_{i,s_i} prod g_xi(theta_j) prod be(V_j; a_j, b_j);
    Example 5.5 coordinate updates for normal mixtures:
    xi_j = (sum pi_ij X_i / sigma^2 + mu/tau^2) /
           (sum pi_ij / sigma^2 + 1/tau^2), eta_j^2 the matching
    precision inverse; responsibilities by softmax. Keys: estimate."""
    xs = _bnp._flat(data)
    n = len(xs)
    M = float(alpha)
    rng = np.random.default_rng(seed)
    lo, hi = min(xs), max(xs)
    span = (hi - lo) or 1.0
    # spread the initial centers across the data range so the
    # coordinatewise ascent has components on every mode
    xi = [lo + span * (j + 0.5) / K for j in range(K)]
    r = [[1.0 / K] * K for _ in range(n)]
    elbo_like = 0.0
    for _ in range(int(n_iter)):
        # responsibilities (softmax over kernel fit; flat weight part)
        for i in range(n):
            logits = [-(xs[i] - xi[j]) ** 2 / (2.0 * sigma * sigma)
                      for j in range(K)]
            mx = max(logits)
            ex = [math.exp(v - mx) for v in logits]
            tot = sum(ex)
            r[i] = [v / tot for v in ex]
        # update (i)-(ii) of Example 5.5
        for j in range(K):
            num = sum(r[i][j] * xs[i] for i in range(n)) \
                / sigma ** 2
            den = sum(r[i][j] for i in range(n)) / sigma ** 2 \
                + 1.0 / tau ** 2
            xi[j] = num / den
        elbo_like = sum(
            r[i][j] * (-(xs[i] - xi[j]) ** 2
                       / (2.0 * sigma * sigma)
                       - math.log(max(r[i][j], 1e-300)))
            for i in range(n) for j in range(K))
    res = RichResult(payload={"estimate": elbo_like,
                              "centers": xi,
                              "method": "mean-field VB for DPM (GvdV 2017 eq. 5.8, Ex 5.5)"})
    return with_describe_pointer(res, "gh_c5_6")


def cheatsheet():
    return "gh_c5_6: Variational algorithm for DPM"


# compact alias per ledger/NAMING.md
ghosalvbdpm = ghosal_vb_dpm
