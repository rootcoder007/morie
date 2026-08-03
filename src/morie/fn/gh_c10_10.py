# morie.fn -- function file (rootcoder007/morie)
"""Random-series Poisson regression.

Implements sec. 10.4.4 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_frs_poireg"]


def ghosal_frs_poireg(n=800, K=3, seed=42):
    """Y | x ~ Poi(exp(f(x))), f a finite random series
    (sec. 10.4.4): gradient MAP recovers the log-rate. Truth
    f0(x) = 1 + 0.8 cos(pi x). Keys: estimate."""
    rng = np.random.default_rng(seed)
    xs = [float(rng.uniform(0, 1)) for _ in range(n)]
    def f0(x):
        return 1.0 + 0.8 * math.cos(math.pi * x)
    def rpois(lam):
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            p *= float(rng.uniform(0, 1))
            if p <= L:
                return k
            k += 1
    ys = [float(rpois(math.exp(f0(x)))) for x in xs]
    def phi(x, k):
        return 1.0 if k == 0 else math.sqrt(2.0) \
            * math.cos(k * math.pi * x)
    beta = [0.0] * K
    for _ in range(200):
        grad = [0.0] * K
        for x, y in zip(xs, ys):
            fx = sum(b * phi(x, k) for k, b in enumerate(beta))
            mu = math.exp(min(fx, 5.0))
            for k in range(K):
                grad[k] += (y - mu) * phi(x, k)
        for k in range(K):
            beta[k] += 0.002 * grad[k] / n * 10.0 - 0.0005 * beta[k]
    err = 0.0
    for j in range(20):
        x = (j + 0.5) / 20
        fx = sum(b * phi(x, k) for k, b in enumerate(beta))
        err += abs(fx - f0(x)) / 20.0
    res = RichResult(payload={"estimate": err, "beta": beta,
                              "method": "series Poisson regression (GvdV 2017 sec. 10.4.4)"})
    return with_describe_pointer(res, "gh_c10_10")


def cheatsheet():
    return "gh_c10_10: Random-series Poisson regression"
