# morie.fn -- function file (rootcoder007/morie)
"""Spline regression contraction.

Implements sec. 9.5.5 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_spline_crt"]


def ghosal_spline_crt(ns=(100, 800, 6400), smoothness=1.0, seed=42):
    """Regression on a spline space of dimension K_n ~ n^{1/(2s+1)}
    attains squared-error rate n^{-2s/(2s+1)} (sec. 9.5.5). Piecewise
    constant fit to f0(x) = sin(2 pi x); risk falls at the expected
    order. Keys: estimate."""
    rng = np.random.default_rng(seed)
    risks = []
    for n in ns:
        K = max(2, int(round(n ** (1.0 / (2 * smoothness + 1.0)))))
        s_ = [0.0] * K
        c_ = [0.0] * K
        for _ in range(n):
            x = float(rng.uniform(0, 1))
            y = math.sin(2.0 * math.pi * x) \
                + 0.3 * float(rng.normal(0, 1))
            b = min(int(x * K), K - 1)
            s_[b] += y
            c_[b] += 1.0
        risk = 0.0
        m = 50
        for i in range(m):
            x = (i + 0.5) / m
            b = min(int(x * K), K - 1)
            fhat = s_[b] / (c_[b] + 1.0)
            risk += (fhat - math.sin(2.0 * math.pi * x)) ** 2 / m
        risks.append(risk)
    rate_hat = math.log(risks[0] / risks[-1]) \
        / math.log(float(ns[-1]) / ns[0])
    res = RichResult(payload={"estimate": rate_hat,
                              "risk_by_n": risks,
                              "expected_exponent":
                                  2 * smoothness / (2 * smoothness
                                                    + 1.0),
                              "method": "spline regression rate (GvdV 2017 sec. 9.5.5)"})
    return with_describe_pointer(res, "gh_c9_10")


def cheatsheet():
    return "gh_c9_10: Spline regression contraction"
