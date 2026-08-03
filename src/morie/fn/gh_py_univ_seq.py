# morie.fn -- function file (rootcoder007/morie)
"""PY universal stick sequence.

Implements sec. 14.4 (V_k law) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_py_universal_sequence"]


def ghosal_py_universal_sequence(d=0.4, theta=2.0, k_list=(1, 5, 20)):
    """(V_k) independent Beta(1 - d, theta + k d): the mean
    E V_k = (1-d)/(1 + theta + k d - d) decreases in k -- late sticks
    take smaller fractions, producing the PY power law (sec. 14.4).
    Keys: estimate."""
    means = [(1.0 - d) / (1.0 - d + theta + k * d) for k in k_list]
    res = RichResult(payload={"estimate": means[0],
                              "mean_by_k": means,
                              "decreasing": all(
                                  means[i + 1] < means[i]
                                  for i in range(len(means) - 1)),
                              "method": "PY stick law (GvdV 2017 sec. 14.4)"})
    return with_describe_pointer(res, "gh_py_univ_seq")


def cheatsheet():
    return "gh_py_univ_seq: PY universal stick sequence"
