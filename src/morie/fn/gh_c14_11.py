# morie.fn -- function file (rootcoder007/morie)
"""Pitman-Yor power law.

Implements sec. 14.4 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_py_powerlaw"]


def ghosal_py_powerlaw(n=5000, d=0.5, theta=1.0, seed=42):
    """E K_n ~ (Gamma(theta+1)/(d Gamma(theta+d))) n^d: PY yields a
    POWER LAW in the number of distinct species, unlike the DP's
    log n (sec. 14.4). Simulated K_n against the constant * n^d law.
    Keys: estimate."""
    rng = np.random.default_rng(seed)
    K = 0
    sizes = []
    for i in range(int(n)):
        u = float(rng.uniform(0, 1)) * (theta + i)
        if u < theta + K * d:
            K += 1
            sizes.append(1.0 - d)
        else:
            acc = theta + K * d
            for t in range(len(sizes)):
                acc += sizes[t]
                if u < acc:
                    sizes[t] += 1.0
                    break
            else:
                sizes[-1] += 1.0
    theory = math.gamma(theta + 1.0) / (d * math.gamma(theta + d)) \
        * float(n) ** d
    res = RichResult(payload={"estimate": float(K),
                              "theory": theory,
                              "ratio": K / theory,
                              "method": "PY power law (GvdV 2017 sec. 14.4)"})
    return with_describe_pointer(res, "gh_c14_11")


def cheatsheet():
    return "gh_c14_11: Pitman-Yor power law"
