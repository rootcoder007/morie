# morie.fn -- function file (rootcoder007/morie)
"""General DPM consistency.

Implements sec. 7.2.2 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dpm_gen_con"]


def ghosal_dpm_gen_con(ns=(10, 100, 3000), alpha=1.0, sd=0.4,
                       seed=42):
    """DPM of normals with well-behaved kernel is consistent at
    mixture truths (sec. 7.2.2, via KL property + entropy). Demo:
    truth N(1, sd^2); the sequential-urn predictive density at the
    truth's center improves with n (Hellinger-type error falls).
    Keys: estimate."""
    def npdf(x, m, s):
        z = (x - m) / s
        return math.exp(-0.5 * z * z) / (s * math.sqrt(2 * math.pi))
    rng = np.random.default_rng(seed)
    s_marg = math.sqrt(1.0 + sd * sd)
    errs = []
    for n in ns:
        data = [1.0 + sd * float(rng.normal(0, 1))
                for _ in range(n)]
        # predictive density at query points via urn mixture
        err = 0.0
        for q in (0.6, 1.0, 1.4):
            pred = alpha / (alpha + n) * npdf(q, 0.0, s_marg) \
                + sum(npdf(q, xj, sd) for xj in data) \
                / (alpha + n)
            err += abs(pred - npdf(q, 1.0, sd))
        errs.append(err / 3.0)
    res = RichResult(payload={"estimate": errs[-1],
                              "error_by_n": errs,
                              "improving": errs[-1] < errs[0],
                              "method": "DPM consistency demo (GvdV 2017 sec. 7.2.2)"})
    return with_describe_pointer(res, "gh_c7_5")


def cheatsheet():
    return "gh_c7_5: General DPM consistency"
