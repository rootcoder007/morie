# morie.fn -- function file (rootcoder007/morie)
"""Parametric vs nonparametric Bayes factor.

Implements sec. 10.5.3 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_param_np_bf"]


def ghosal_param_np_bf(n=1500, parametric_truth=True, seed=42):
    """BF ~ exp(-n KL(P0; P-hat)) / Pi(KL ball eps_n): the parametric
    model wins by prior-concentration when it holds; the
    nonparametric wins otherwise (sec. 10.5.3). Multinomial demo:
    H0 = uniform(4), H1 = full Dirichlet simplex; exact evidences.
    Keys: estimate."""
    rng = np.random.default_rng(seed)
    p0 = [0.25] * 4 if parametric_truth else [0.4, 0.3, 0.2, 0.1]
    counts = [0] * 4
    for _ in range(n):
        u = float(rng.uniform(0, 1))
        acc = 0.0
        for i, q in enumerate(p0):
            acc += q
            if u <= acc:
                counts[i] += 1
                break
    # H0 evidence: all cells 1/4
    l0 = sum(c * math.log(0.25) for c in counts)
    # H1 evidence: Dirichlet(1,1,1,1) marginal =
    # Gamma(4) prod Gamma(1+c) / Gamma(4+n)
    l1 = math.lgamma(4.0) - math.lgamma(4.0 + n) \
        + sum(math.lgamma(1.0 + c) for c in counts)
    log_bf_np = l1 - l0
    res = RichResult(payload={"estimate": log_bf_np,
                              "nonparametric_wins": log_bf_np > 0,
                              "method": "parametric-vs-NP Bayes factor (GvdV 2017 sec. 10.5.3)"})
    return with_describe_pointer(res, "gh_c10_14")


def cheatsheet():
    return "gh_c10_14: Parametric vs nonparametric Bayes factor"
