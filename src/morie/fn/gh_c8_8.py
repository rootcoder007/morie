# morie.fn -- function file (rootcoder007/morie)
"""Gaussian regression contraction.

Implements sec. 8.3.2 (conjugate series computation, Ex 8.6 form) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_gauss_reg_crt"]


def ghosal_gauss_reg_crt(s_true=2.0, alpha_prior=2.0,
                         ns=(100, 1000, 10000)):
    """Fixed-design regression with a conjugate series prior: the
    coordinatewise posterior (8.1 analog) gives total posterior risk
    sum_i [bias_i^2 + var_i]; it decays like n^{-2 min(a,s)/(2a+1)}
    (sec. 8.3.2). Truth theta_{0,i} = i^{-s-1/2}. Keys: estimate."""
    a = float(alpha_prior)
    risks = []
    for n in ns:
        risk = 0.0
        i = 1
        while i < 5000:
            th0 = float(i) ** (-(s_true + 0.5))
            lam = float(i) ** (2.0 * a + 1.0)
            bias = th0 * lam / (n + lam)
            var = 1.0 / (n + lam)
            risk += bias * bias + var
            i += 1
        risks.append(risk)
    rate_hat = math.log(risks[0] / risks[-1]) \
        / math.log(float(ns[-1]) / ns[0])
    expect = 2.0 * min(a, s_true) / (2.0 * a + 1.0)
    res = RichResult(payload={"estimate": rate_hat,
                              "risk_by_n": risks,
                              "expected_exponent": expect,
                              "method": "conjugate regression risk (GvdV 2017 sec. 8.3.2, eq. 8.1)"})
    return with_describe_pointer(res, "gh_c8_8")


def cheatsheet():
    return "gh_c8_8: Gaussian regression contraction"
