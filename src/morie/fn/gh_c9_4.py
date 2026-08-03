# morie.fn -- function file (rootcoder007/morie)
"""DPM-of-normals contraction.

Implements sec. 9.4 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dpm_norm_crt"]


def ghosal_dpm_norm_crt(ns=(10, 100, 3000), alpha=1.0, sd=0.3,
                        seed=42):
    """DPM of normals attains the near-parametric rate
    n^{-s/(2s+1)} (log n)^t for smooth truths (sec. 9.4). Sequential
    urn predictive density error at query points falls with n.
    Keys: estimate."""
    def npdf(x, m, s):
        z = (x - m) / s
        return math.exp(-0.5 * z * z) / (s * math.sqrt(2 * math.pi))
    rng = np.random.default_rng(seed)
    s_marg = math.sqrt(1.0 + sd * sd)
    errs = []
    for n in ns:
        data = [0.5 + sd * float(rng.normal(0, 1))
                for _ in range(n)]
        err = 0.0
        for q in (0.2, 0.5, 0.8):
            pred = alpha / (alpha + n) * npdf(q, 0.0, s_marg) \
                + sum(npdf(q, xj, sd) for xj in data) / (alpha + n)
            err += abs(pred - npdf(q, 0.5, sd)) / 3.0
        errs.append(err)
    res = RichResult(payload={"estimate": errs[-1],
                              "err_by_n": errs,
                              "improving": errs[-1] < errs[0],
                              "method": "DPM-of-normals rate (GvdV 2017 sec. 9.4)"})
    return with_describe_pointer(res, "gh_c9_4")


def cheatsheet():
    return "gh_c9_4: DPM-of-normals contraction"
