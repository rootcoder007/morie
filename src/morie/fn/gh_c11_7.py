# morie.fn -- function file (rootcoder007/morie)
"""Riemann-Liouville process.

Implements Example 11.6, eq. (11.2) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_rl_process"]


def ghosal_rl_process(alpha=0.75, n_grid=200, n_sim=300, seed=42):
    """R_t^alpha = Gamma(alpha + 1/2)^{-1} int_0^t (t-s)^{alpha-1/2}
    dB_s (eq. 11.2): Gaussian, self-similar of index alpha, so
    var(R_t) proportional to t^{2 alpha}. Discretized-integral check
    of the variance-growth exponent. Keys: estimate."""
    rng = np.random.default_rng(seed)
    g = math.gamma(alpha + 0.5)
    t1_idx, t2_idx = n_grid // 4, n_grid
    v1 = v2 = 0.0
    for _ in range(n_sim):
        dB = [float(rng.normal(0, 1)) / math.sqrt(n_grid)
              for _ in range(n_grid)]
        for t_idx, tag in ((t1_idx, 1), (t2_idx, 2)):
            t = t_idx / n_grid
            r = sum((t - (j + 0.5) / n_grid) ** (alpha - 0.5) * dB[j]
                    for j in range(t_idx)) / g
            if tag == 1:
                v1 += r * r / n_sim
            else:
                v2 += r * r / n_sim
    growth = math.log(v2 / v1) / math.log(4.0)
    res = RichResult(payload={"estimate": growth,
                              "expected": 2.0 * alpha,
                              "gap": abs(growth - 2.0 * alpha),
                              "method": "Riemann-Liouville process (GvdV 2017 eq. 11.2)"})
    return with_describe_pointer(res, "gh_c11_7")


def cheatsheet():
    return "gh_c11_7: Riemann-Liouville process"
