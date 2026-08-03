# morie.fn -- function file (rootcoder007/morie)
"""Random measure with consistent finite-dimensional laws.

Implements sec. 3.1 (Kolmogorov construction) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_random_measure_def"]


def ghosal_random_measure_def(x, n_atoms=50, seed=42):
    """A random probability measure realized as normalized random
    weights on atoms (GvdV 2017 sec. 3.1): additivity of the induced
    set function is checked explicitly, the Kolmogorov-consistency
    property the section formalizes."""
    xs = _bnp._flat(x)
    rng = np.random.default_rng(seed)
    atoms = [float(v) for v in rng.uniform(0, 1, n_atoms)._flat()]
    w = _bnp.normalize_weights(
        [float(v) for v in rng.gamma(1.0, 1.0, n_atoms)._flat()])
    def P(a, b):
        return sum(wi for wi, t in zip(w, atoms) if a <= t < b)
    additivity_gap = abs(P(0, 1) - (P(0, 0.4) + P(0.4, 1)))
    est = P(0, 0.5)
    res = RichResult(payload={"estimate": est,
                              "additivity_gap": additivity_gap,
                              "total_mass": P(0, 1),
                              "method": "random measure via normalized atoms (GvdV 2017 sec. 3.1)"})
    return with_describe_pointer(res, "gh_c3_1")


def cheatsheet():
    return "gh_c3_1: Random measure with consistent finite-dimensional laws"
