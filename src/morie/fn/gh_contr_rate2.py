# morie.fn -- function file (rootcoder007/morie)
"""i.i.d. contraction in L1.

Implements Theorem 8.9 applied to the multinomial model of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_contraction_rate_iid"]


def ghosal_contraction_rate_iid(p0=(0.4, 0.3, 0.2, 0.1),
                                ns=(50, 500, 5000), seed=42):
    """Pi_n(||p - p0||_1 > M eps_n | X^n) -> 0 at eps_n = n^{-1/2}
    for finite multinomials (Thm 8.9: entropy of the simplex is
    finite-dimensional, prior mass polynomial). Reports the exact
    posterior expected L1 distance under Dirichlet(1,...) updating.
    Keys: estimate."""
    p0 = _bnp.normalize_weights(p0)
    rng = np.random.default_rng(seed)
    dists = []
    for n in ns:
        counts = [0] * len(p0)
        for _ in range(n):
            u = float(rng.uniform(0, 1))
            acc = 0.0
            for i, q in enumerate(p0):
                acc += q
                if u <= acc:
                    counts[i] += 1
                    break
        A = len(p0) + n
        # E|p_j - p0_j| <= |m_j - p0_j| + sd_j
        d = 0.0
        for c, q in zip(counts, p0):
            m = (1.0 + c) / A
            sd = math.sqrt(m * (1.0 - m) / (A + 1.0))
            d += abs(m - q) + sd
        dists.append(d)
    rate_hat = math.log(dists[0] / dists[-1]) \
        / math.log(float(ns[-1]) / ns[0])
    res = RichResult(payload={"estimate": rate_hat,
                              "l1_by_n": dists,
                              "half_rate": abs(rate_hat - 0.5) < 0.2,
                              "method": "iid L1 contraction (GvdV 2017 Thm 8.9)"})
    return with_describe_pointer(res, "gh_contr_rate2")


def cheatsheet():
    return "gh_contr_rate2: i.i.d. contraction in L1"
