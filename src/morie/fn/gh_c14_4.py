# morie.fn -- function file (rootcoder007/morie)
"""Chinese restaurant franchise.

Implements sec. 14.1.2 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_crf_def"]


def ghosal_crf_def(n_per_rest=(60, 60), alpha=2.0, gamma=2.0,
                   seed=42):
    """CRF: per-restaurant CRPs whose new tables order dishes from a
    global CRP -- G_j ~ DP(alpha, G0), G0 ~ DP(gamma, H)
    (sec. 14.1.2): dishes are SHARED across restaurants.
    Keys: estimate."""
    rng = np.random.default_rng(seed)
    global_dishes = []
    shared = 0
    per_rest_dishes = []
    for n in n_per_rest:
        tables = []
        dish_of_table = []
        for i in range(int(n)):
            u = float(rng.uniform(0, 1)) * (alpha + i)
            if u < alpha:
                # new table: draw dish from global CRP
                m = sum(global_dishes)
                ug = float(rng.uniform(0, 1)) * (gamma + m)
                if ug < gamma:
                    global_dishes.append(1)
                    dish_of_table.append(len(global_dishes) - 1)
                else:
                    acc = gamma
                    for d in range(len(global_dishes)):
                        acc += global_dishes[d]
                        if ug < acc:
                            global_dishes[d] += 1
                            dish_of_table.append(d)
                            break
                tables.append(1)
            else:
                acc = alpha
                for t in range(len(tables)):
                    acc += tables[t]
                    if u < acc:
                        tables[t] += 1
                        break
        per_rest_dishes.append(set(dish_of_table))
    shared = len(per_rest_dishes[0] & per_rest_dishes[1])
    res = RichResult(payload={"estimate": float(shared),
                              "n_global_dishes": len(global_dishes),
                              "dishes_shared": shared > 0,
                              "method": "Chinese restaurant franchise (GvdV 2017 sec. 14.1.2)"})
    return with_describe_pointer(res, "gh_c14_4")


def cheatsheet():
    return "gh_c14_4: Chinese restaurant franchise"


# compact alias per ledger/NAMING.md
ghosalcrfdef = ghosal_crf_def
