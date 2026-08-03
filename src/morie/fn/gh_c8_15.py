# morie.fn -- function file (rootcoder007/morie)
"""Alpha-posterior contraction.

Implements sec. 8.6 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_alpha_pst_crt"]


def ghosal_alpha_pst_crt(theta0=0.5, alpha=0.5, ns=(100, 10000),
                         seed=42):
    """The alpha-posterior contracts at the rate determined by the
    prior concentration ALONE for alpha < 1 (sec. 8.6): no entropy
    condition needed. Beta-Bernoulli alpha-posterior variance ~
    1/(alpha n): same n^{-1/2} rate as the full posterior.
    Keys: estimate."""
    rng = np.random.default_rng(seed)
    vars_ = []
    for n in ns:
        S = sum(1 for _ in range(n)
                if float(rng.uniform(0, 1)) < theta0)
        a_p, b_p = 1.0 + alpha * S, 1.0 + alpha * (n - S)
        vars_.append(a_p * b_p / ((a_p + b_p) ** 2
                                  * (a_p + b_p + 1.0)))
    rate_hat = math.log(vars_[0] / vars_[-1]) \
        / math.log(float(ns[-1]) / ns[0])
    res = RichResult(payload={"estimate": rate_hat,
                              "var_by_n": vars_,
                              "parametric_rate": abs(rate_hat - 1.0)
                              < 0.15,
                              "method": "alpha-posterior contraction (GvdV 2017 sec. 8.6)"})
    return with_describe_pointer(res, "gh_c8_15")


def cheatsheet():
    return "gh_c8_15: Alpha-posterior contraction"
