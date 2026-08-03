# morie.fn -- function file (rootcoder007/morie)
"""Alpha-posterior (tempered likelihood).

Implements sec. 6.8.5 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_alpha_post"]


def ghosal_alpha_post(successes, n, alpha=0.5, a=1.0, b=1.0):
    """pi_alpha(theta | X^n) propto pi(theta) L_n(theta)^alpha,
    0 < alpha < 1 (sec. 6.8.5): for Beta-Bernoulli the alpha-posterior
    is Beta(a + alpha S, b + alpha(n-S)) -- flatter than the full
    posterior, same center in the limit. Keys: estimate."""
    S = float(successes)
    n = float(n)
    al = float(alpha)
    a_post, b_post = a + al * S, b + al * (n - S)
    mean = a_post / (a_post + b_post)
    var = a_post * b_post / ((a_post + b_post) ** 2
                             * (a_post + b_post + 1.0))
    full_var = (a + S) * (b + n - S) / ((a + b + n) ** 2
                                        * (a + b + n + 1.0))
    res = RichResult(payload={"estimate": mean,
                              "alpha_posterior": [a_post, b_post],
                              "variance": var,
                              "wider_than_full": var > full_var,
                              "method": "alpha-posterior (GvdV 2017 sec. 6.8.5)"})
    return with_describe_pointer(res, "gh_c6_16")


def cheatsheet():
    return "gh_c6_16: Alpha-posterior (tempered likelihood)"
