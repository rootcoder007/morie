# morie.fn -- function file (rootcoder007/morie)
"""Fractional Brownian motion prior.

Implements Example 11.9, eq. (11.6) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_fbm_prior"]


def ghosal_fbm_prior(H=0.7, ts=(0.25, 0.5, 0.75)):
    """fBm: E(W_s W_t) = (s^{2H} + t^{2H} - |t - s|^{2H})/2
    (eq. 11.6): evaluates the kernel matrix and verifies
    E W_t^2 = t^{2H} plus positive-definiteness on the grid
    (Cholesky-style leading minors positive). Keys: estimate."""
    H = float(H)
    def K(s, t):
        return 0.5 * (s ** (2 * H) + t ** (2 * H)
                      - abs(t - s) ** (2 * H))
    G = [[K(a, b) for b in ts] for a in ts]
    var_gap = max(abs(G[i][i] - ts[i] ** (2 * H))
                  for i in range(len(ts)))
    # leading principal minors via Gaussian elimination
    m = [row[:] for row in G]
    minors = []
    det = 1.0
    for i in range(len(ts)):
        det *= m[i][i]
        minors.append(det)
        for r in range(i + 1, len(ts)):
            f = m[r][i] / m[i][i]
            for c in range(len(ts)):
                m[r][c] -= f * m[i][c]
    res = RichResult(payload={"estimate": G[0][0],
                              "kernel": G,
                              "var_gap": var_gap,
                              "positive_definite": all(v > 0
                                                       for v in
                                                       minors),
                              "method": "fBm covariance (GvdV 2017 eq. 11.6)"})
    return with_describe_pointer(res, "gh_c11_8")


def cheatsheet():
    return "gh_c11_8: Fractional Brownian motion prior"


# compact alias per ledger/NAMING.md
ghosalfbmprior = ghosal_fbm_prior
