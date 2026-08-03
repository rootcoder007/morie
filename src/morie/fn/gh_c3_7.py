# morie.fn -- function file (rootcoder007/morie)
"""Prior via random rectangular partitions.

Implements sec. 3.4.3 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_rect_partition"]


def ghosal_rect_partition(x, depth=5, seed=42):
    """Random measure by recursively splitting [0,1] at random points
    and assigning random mass fractions (GvdV 2017 sec. 3.4.3):
    returns cell boundaries and masses summing to 1."""
    rng = np.random.default_rng(seed)
    cells = [(0.0, 1.0, 1.0)]
    for _ in range(depth):
        nxt = []
        for lo, hi, m in cells:
            cut = lo + (hi - lo) * (0.25 + 0.5 * float(
                rng.uniform(0, 1)))
            frac = float(rng.beta(1.0, 1.0))
            nxt.append((lo, cut, m * frac))
            nxt.append((cut, hi, m * (1.0 - frac)))
        cells = nxt
    total = sum(m for _, _, m in cells)
    mean = sum(0.5 * (lo + hi) * m for lo, hi, m in cells)
    res = RichResult(payload={"estimate": mean, "n_cells": len(cells),
                              "total_mass": total,
                              "method": "random rectangular partition prior (GvdV 2017 sec. 3.4.3)"})
    return with_describe_pointer(res, "gh_c3_7")


def cheatsheet():
    return "gh_c3_7: Prior via random rectangular partitions"
