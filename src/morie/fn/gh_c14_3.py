# morie.fn -- function file (rootcoder007/morie)
"""Chinese restaurant process.

Implements sec. 14.1.1 (eq. 4.13) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_crp_def"]


def ghosal_crp_def(n=100, alpha=2.0, seed=42):
    """P(join table k) = n_k/(alpha + n), new table with
    alpha/(alpha + n) (eq. 4.13): simulates the CRP; E K_n =
    sum alpha/(alpha + i - 1) (Prop 4.8). Keys: estimate."""
    rng = np.random.default_rng(seed)
    tables = []
    for i in range(int(n)):
        u = float(rng.uniform(0, 1)) * (alpha + i)
        if u < alpha:
            tables.append(1)
        else:
            acc = alpha
            for t in range(len(tables)):
                acc += tables[t]
                if u < acc:
                    tables[t] += 1
                    break
            else:
                tables[-1] += 1
    EK = sum(alpha / (alpha + i - 1.0) for i in range(1, int(n) + 1))
    res = RichResult(payload={"estimate": float(len(tables)),
                              "expected_K_n": EK,
                              "sizes": sorted(tables, reverse=True),
                              "total_seated": sum(tables),
                              "method": "CRP (GvdV 2017 eq. 4.13)"})
    return with_describe_pointer(res, "gh_c14_3")


def cheatsheet():
    return "gh_c14_3: Chinese restaurant process"


# compact alias per ledger/NAMING.md
ghosalcrpdef = ghosal_crp_def
