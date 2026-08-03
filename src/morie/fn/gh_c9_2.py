# morie.fn -- function file (rootcoder007/morie)
"""DP posterior CDF contraction.

Implements sec. 9.2 (parametric-rate DP, eq. 4.11-4.12 machinery) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dp_disc_crt"]


def ghosal_dp_disc_crt(ns=(100, 400, 1600), alpha=2.0, seed=42):
    """The DP posterior mean CDF satisfies sup-error O_P(n^{-1/2})
    (sec. 9.2, via eq. 4.11-4.12): the Kolmogorov error against a
    uniform truth falls at the parametric rate. Keys: estimate."""
    rng = np.random.default_rng(seed)
    errs = []
    for n in ns:
        # average the sup-error over replications to tame MC noise
        reps = 12
        avg = 0.0
        for _ in range(reps):
            data = [float(rng.uniform(0, 1)) for _ in range(n)]
            err = 0.0
            for j in range(1, 20):
                t = j / 20.0
                emp = sum(1 for v in data if v <= t)
                post = (alpha * t + emp) / (alpha + n)
                err = max(err, abs(post - t))
            avg += err / reps
        errs.append(avg)
    rate_hat = math.log(errs[0] / errs[-1]) \
        / math.log(float(ns[-1]) / ns[0])
    res = RichResult(payload={"estimate": rate_hat,
                              "sup_err_by_n": errs,
                              "near_half": abs(rate_hat - 0.5) < 0.3,
                              "method": "DP CDF contraction (GvdV 2017 sec. 9.2)"})
    return with_describe_pointer(res, "gh_c9_2")


def cheatsheet():
    return "gh_c9_2: DP posterior CDF contraction"
