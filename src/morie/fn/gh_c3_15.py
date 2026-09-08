# morie.fn -- function file (rootcoder007/morie)
"""Partially specified Pólya tree.

Implements sec. 3.7.3 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_partspec_pt"]


def ghosal_partspec_pt(x, levels=(2, 4), a_scale=1.0, seed=42):
    """PT specified only at selected levels m1 < m2 < ... with the
    rest marginalized (GvdV 2017 sec. 3.7.3): between specified
    levels the split is the prior mean 1/2, so cell masses only
    change at the specified levels."""
    rng = np.random.default_rng(seed)
    xs = _bnp._flat(x)
    x0 = xs[0] if xs else 0.3
    depth = max(levels)
    bits = _bnp._bits(x0, depth)
    mass = 1.0
    for m, b in enumerate(bits, start=1):
        if m in set(int(v) for v in levels):
            a = a_scale * m * m
            V0 = float(rng.beta(a, a))
        else:
            V0 = 0.5                     # marginalized level
        mass *= V0 if b == 0 else (1.0 - V0)
    res = RichResult(payload={"estimate": mass * 2.0 ** depth,
                              "cell_mass": mass,
                              "specified_levels": list(levels),
                              "method": "partially specified PT (GvdV 2017 sec. 3.7.3)"})
    return with_describe_pointer(res, "gh_c3_15")


def cheatsheet():
    return "gh_c3_15: Partially specified Pólya tree"
