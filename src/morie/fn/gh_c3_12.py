# morie.fn -- function file (rootcoder007/morie)
"""Pólya tree process definition.

Implements sec. 3.7 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_polya_tree_def"]


def ghosal_polya_tree_def(x, depth=8, a_scale=1.0, seed=42):
    """PT: independent Y_{e0|e} ~ Beta(alpha_e0, alpha_e1) splitting
    variables, set masses as products down the tree (GvdV 2017
    sec. 3.7); canonical choice alpha at level m = a_scale * m^2 keeps
    the density a.s. well behaved."""
    rng = np.random.default_rng(seed)
    xs = _bnp._flat(x)
    x0 = xs[0] if xs else 0.3
    bits = _bnp._bits(x0, depth)
    mass = 1.0
    for m, b in enumerate(bits, start=1):
        a = a_scale * m * m
        V0 = float(rng.beta(a, a))
        mass *= V0 if b == 0 else (1.0 - V0)
    density = mass * 2.0 ** depth
    res = RichResult(payload={"estimate": density,
                              "cell_mass": mass, "depth": depth,
                              "method": "Polya tree draw (GvdV 2017 sec. 3.7)"})
    return with_describe_pointer(res, "gh_c3_12")


def cheatsheet():
    return "gh_c3_12: Pólya tree process definition"
