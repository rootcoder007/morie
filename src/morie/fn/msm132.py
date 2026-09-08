# morie.fn -- function file (rootcoder007/morie)
"""Deep arc-cosine kernel.

Implements eq. (8.5) p.266 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries the previous chapter's topic label;
chapter 8 is Reproducing Kernel Hilbert Spaces regression, and the
canonical name below reflects that.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_categorical_count_eq_8_5", "mvsml_arccos_kernel_deep"]


def mvsml_categorical_count_eq_8_5(X, Z=None, depth=2, normalize_median=False):
    """AK^(l+1)(x_i,x_j) = (1/pi)[AK^l(x_i,x_i) AK^l(x_j,x_j)]^(1/2)
    J(theta^l_ij) with theta^l_ij = arccos{AK^l(x_i,x_j)
    [AK^l(x_i,x_i) AK^l(x_j,x_j)]^(-1/2)} (eq. 8.5): repeating the
    interior product l times emulates l hidden layers, which is what
    makes this kernel behave like a deep network.  No bandwidth
    parameter is required; only the number of layers.
    Keys: estimate."""
    K = _gp.arccos_kernel(X, Z=Z, depth=depth,
                          normalize_median=normalize_median)
    ok, lam = _gp.is_positive_semidefinite(K) if Z is None \
        else (None, None)
    res = RichResult(payload={"estimate": K[0][0], "kernel": K,
                              "depth": int(depth),
                              "positive_semidefinite": ok,
                              "method": "deep arc-cosine kernel (MVSML 2022 eq. 8.5)"})
    return with_describe_pointer(res, "msm132")


mvsml_arccos_kernel_deep = mvsml_categorical_count_eq_8_5


def cheatsheet():
    return "msm132: Deep arc-cosine kernel"
