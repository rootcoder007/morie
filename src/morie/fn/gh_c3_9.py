# morie.fn -- function file (rootcoder007/morie)
"""Prior through random quantiles.

Implements sec. 3.4.5 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_quantile_prior"]


def ghosal_quantile_prior(x, n_knots=15, seed=42):
    """Construct G through its quantile function Q(u) = F^{-1}(u)
    (GvdV 2017 sec. 3.4.5): random increasing knots interpolated
    monotonically; the induced CDF is uniform-consistent with Q."""
    rng = np.random.default_rng(seed)
    us = [j / (n_knots + 1.0) for j in range(1, n_knots + 1)]
    incs = [float(rng.gamma(1.0, 1.0)) for _ in range(n_knots + 1)]
    tot = sum(incs)
    qs = []
    acc = 0.0
    for i in range(n_knots):
        acc += incs[i]
        qs.append(acc / tot)
    monotone = all(qs[i] < qs[i + 1] for i in range(len(qs) - 1))
    med = qs[n_knots // 2]
    res = RichResult(payload={"estimate": med, "u": us, "Q": qs,
                              "monotone": monotone,
                              "method": "random quantile-function prior (GvdV 2017 sec. 3.4.5)"})
    return with_describe_pointer(res, "gh_c3_9")


def cheatsheet():
    return "gh_c3_9: Prior through random quantiles"
