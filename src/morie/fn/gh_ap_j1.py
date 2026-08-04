# morie.fn -- function file (rootcoder007/morie)
"""Lévy-Itô decomposition.

Implements Appendix J of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_levy_ito"]


def ghosal_levy_ito(fixed_atoms=(0.5,), atom_masses=(0.2,),
                    n_random_jumps=200, seed=42):
    """M = M_fixed + M_atomic + M_diffuse: every CRM decomposes into
    fixed atoms, random atoms (Poisson), and a deterministic drift
    (App J). Assembles the pieces and reports the total mass split.
    Keys: estimate."""
    rng = np.random.default_rng(seed)
    fixed = sum(_bnp._flat(atom_masses))
    random_part = sum(float(rng.gamma(1.0 / n_random_jumps, 1.0))
                      for _ in range(n_random_jumps))
    total = fixed + random_part
    res = RichResult(payload={"estimate": total,
                              "fixed_mass": fixed,
                              "poisson_mass": random_part,
                              "method": "Levy-Ito decomposition (GvdV 2017 App J)"})
    return with_describe_pointer(res, "gh_ap_j1")


def cheatsheet():
    return "gh_ap_j1: Lévy-Itô decomposition"


# compact alias per ledger/NAMING.md
ghosallevyito = ghosal_levy_ito
