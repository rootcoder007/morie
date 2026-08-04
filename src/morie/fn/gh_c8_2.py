# morie.fn -- function file (rootcoder007/morie)
"""Basic contraction-rate theorem.

Implements Theorem 8.9, conditions (8.4)-(8.6) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ggv_thm"]


def ghosal_ggv_thm(n, eps_n, log_prior_mass_B2, log_entropy,
                   log_sieve_tail_mass, C=1.0):
    """Thm 8.9: eps_n is a contraction rate if (i)
    Pi(B_2(p0, eps_n)) >= e^{-C n eps_n^2}, (ii)
    log N(xi eps_n, P_n1, d) <= n eps_n^2, (iii)
    Pi(P_n2) <= e^{-(C+4) n eps_n^2}. Checks the three inequalities.
    Keys: estimate."""
    ne2 = float(n) * float(eps_n) ** 2
    c_i = float(log_prior_mass_B2) >= -C * ne2
    c_ii = float(log_entropy) <= ne2
    c_iii = float(log_sieve_tail_mass) <= -(C + 4.0) * ne2
    ok = c_i and c_ii and c_iii
    res = RichResult(payload={"estimate": float(eps_n) if ok
                              else float("nan"),
                              "n_eps2": ne2,
                              "conditions": [c_i, c_ii, c_iii],
                              "rate_certified": ok,
                              "method": "basic rate theorem (GvdV 2017 Thm 8.9)"})
    return with_describe_pointer(res, "gh_c8_2")


def cheatsheet():
    return "gh_c8_2: Basic contraction-rate theorem"


# compact alias per ledger/NAMING.md
ghosalggvthm = ghosal_ggv_thm
