# morie.fn -- function file (rootcoder007/morie)
"""Random-series binary regression.

Implements sec. 10.4.3 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_frs_binreg"]


def ghosal_frs_binreg(n=800, K=4, seed=42):
    """P(Y=1|x) = Phi(f(x)), f a finite random series: adaptive rate
    (sec. 10.4.3). Damped-Newton MAP for a probit-type fit (logistic
    link surrogate with matched slope); classification recovers the
    monotone truth. Keys: estimate."""
    rng = np.random.default_rng(seed)
    xs = [float(rng.uniform(0, 1)) for _ in range(n)]
    def F0(x):
        return 1.0 / (1.0 + math.exp(-4.0 * (x - 0.5)))
    ys = [1.0 if float(rng.uniform(0, 1)) < F0(x) else 0.0
          for x in xs]
    def phi(x, k):
        return 1.0 if k == 0 else math.sqrt(2.0) \
            * math.cos(k * math.pi * x)
    beta = [0.0] * K
    for _ in range(80):
        grad = [0.0] * K
        for x, y in zip(xs, ys):
            fx = sum(b * phi(x, k) for k, b in enumerate(beta))
            p = 1.0 / (1.0 + math.exp(-fx))
            for k in range(K):
                grad[k] += (y - p) * phi(x, k)
        for k in range(K):
            beta[k] += 0.02 * grad[k] / n * 4.0 - 0.001 * beta[k]
    # accuracy at the extremes
    def fit(x):
        fx = sum(b * phi(x, k) for k, b in enumerate(beta))
        return 1.0 / (1.0 + math.exp(-fx))
    acc = (fit(0.9) > 0.5) and (fit(0.1) < 0.5)
    err = abs(fit(0.5) - 0.5)
    res = RichResult(payload={"estimate": err,
                              "orders_correctly": acc,
                              "p_low_high": [fit(0.1), fit(0.9)],
                              "method": "series binary regression (GvdV 2017 sec. 10.4.3)"})
    return with_describe_pointer(res, "gh_c10_9")


def cheatsheet():
    return "gh_c10_9: Random-series binary regression"
