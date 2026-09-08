# morie.fn -- function file (rootcoder007/morie)
"""Prior through a distribution on a dense subset.

Implements sec. 3.4.1 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dense_subset_prior"]


def ghosal_dense_subset_prior(x, n_atoms=200, seed=42):
    """G = sum_k w_k delta_{X_k} with atoms enumerating a dense
    sequence (dyadic rationals here) -- weak support is everything
    (GvdV 2017 sec. 3.4.1)."""
    rng = np.random.default_rng(seed)
    atoms = []
    level = 1
    while len(atoms) < n_atoms:
        atoms += [i / 2.0 ** level for i in range(1, 2 ** level, 2)]
        level += 1
    atoms = atoms[:n_atoms]
    w = _bnp.normalize_weights(
        [2.0 ** (-(j + 1)) * float(rng.gamma(1.0, 1.0))
         for j in range(n_atoms)])
    mean = sum(wi * t for wi, t in zip(w, atoms))
    res = RichResult(payload={"estimate": mean, "atoms": atoms[:16],
                              "weights_head": w[:16],
                              "method": "dense-subset atom prior (GvdV 2017 sec. 3.4.1)"})
    return with_describe_pointer(res, "gh_c3_6")


def cheatsheet():
    return "gh_c3_6: Prior through a distribution on a dense subset"
