# morie.fn -- function file (rootcoder007/morie)
"""Stick-breaking random measure.

Implements sec. 3.3.2, eq. (3.2) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_stick_break_def"]


def ghosal_stick_break_def(x, n_terms=100, a=1.0, b=1.0, seed=42):
    """G = sum_k p_k delta_{theta_k} with p from eq. (3.2) stick
    breaking, V_k ~ Beta(a, b), theta_k from the base measure."""
    rng = np.random.default_rng(seed)
    V = [float(rng.beta(a, b)) for _ in range(n_terms)]
    p = _bnp.stick_breaking(V)
    th = [float(v) for v in rng.uniform(0, 1, n_terms)._flat()]
    mass = sum(p)
    mean = sum(pi * t for pi, t in zip(p, th))
    res = RichResult(payload={"estimate": mean, "weights": p,
                              "atoms": th, "total_mass": mass,
                              "method": "stick-breaking measure (GvdV 2017 eq. 3.2)"})
    return with_describe_pointer(res, "gh_c3_4")


def cheatsheet():
    return "gh_c3_4: Stick-breaking random measure"
