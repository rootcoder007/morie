# morie.fn -- function file (rootcoder007/morie)
"""Distribution of the DP median.

Implements Theorem 4.25 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dp_median"]


def _beta_pdf_log(u, a, b):
    return (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
            + (a - 1.0) * math.log(u) + (b - 1.0) * math.log(1.0 - u))


def ghosal_dp_median(G_x, alpha, n_grid=4000):
    """H(x) = int_{1/2}^1 Be(u; M G(x), M(1-G(x))) du =
    P(Be(MG(x), M(1-G(x))) >= 1/2) (Theorem 4.25): the exact CDF of
    any median of F ~ DP(MG). Composite-midpoint quadrature.
    Keys: estimate."""
    g = float(_bnp._flat(G_x)[0])
    M = float(alpha)
    a, b = M * g, M * (1.0 - g)
    if a <= 0 or b <= 0:
        raise ValueError("need 0 < G(x) < 1 and alpha > 0")
    tot = 0.0
    h = 0.5 / n_grid
    for i in range(n_grid):
        u = 0.5 + (i + 0.5) * h
        if u >= 1.0:
            continue
        tot += math.exp(_beta_pdf_log(u, a, b)) * h
    res = RichResult(payload={"estimate": tot,
                              "beta_params": [a, b],
                              "method": "median-Dirichlet CDF (GvdV 2017 Thm 4.25)"})
    return with_describe_pointer(res, "gh_c4_17")


def cheatsheet():
    return "gh_c4_17: Distribution of the DP median"


# compact alias per ledger/NAMING.md
ghosaldpmedian = ghosal_dp_median
