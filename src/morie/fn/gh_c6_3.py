# morie.fn -- function file (rootcoder007/morie)
"""Doob almost-everywhere consistency.

Implements Theorem 6.9 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_doob_consist"]


def ghosal_doob_consist(theta0=0.4, n=1500, seed=42):
    """Doob: E(f(theta) | X^n) -> f(theta) a.s. [Pi-a.e. theta]
    (Thm 6.9, via martingale convergence). Demonstrated with
    f = identity in the Beta-Bernoulli filtration: the posterior
    mean path converges to theta0. Keys: estimate."""
    rng = np.random.default_rng(seed)
    S = 0
    path = []
    for i in range(1, n + 1):
        S += 1 if float(rng.uniform(0, 1)) < theta0 else 0
        if i % (n // 10) == 0:
            path.append((1.0 + S) / (2.0 + i))
    res = RichResult(payload={"estimate": path[-1],
                              "posterior_mean_path": path,
                              "final_error": abs(path[-1] - theta0),
                              "method": "Doob martingale consistency (GvdV 2017 Thm 6.9)"})
    return with_describe_pointer(res, "gh_c6_3")


def cheatsheet():
    return "gh_c6_3: Doob almost-everywhere consistency"
