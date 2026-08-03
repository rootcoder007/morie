# morie.fn -- function file (rootcoder007/morie)
"""Arc-cosine kernel with one hidden layer.

Implements eq. (8.4) p.265 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries the previous chapter's topic label;
chapter 8 is Reproducing Kernel Hilbert Spaces regression, and the
canonical name below reflects that.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_categorical_count_eq_8_4", "mvsml_arccos_kernel"]


def mvsml_categorical_count_eq_8_4(X, Z=None, normalize_median=False):
    """AK^1(x_i, x_j) = (1/pi) ||x_i|| ||x_j|| J(theta_ij) with
    theta_ij = arccos(x_i'x_j/(||x_i|| ||x_j||)) and
    J(theta) = sin(theta) + (pi - theta) cos(theta) (eq. 8.4).
    Positive semi-definite, norm preserving -- AK(x, x) = ||x||^2 and
    AK(x, -x) = 0 -- and equivalent to a single-hidden-layer network
    with a ramp activation.  Unlike the Gaussian kernel its diagonal
    is heterogeneous, so it expresses per-individual genetic variance.
    Keys: estimate."""
    K = _gp.arccos_kernel(X, Z=Z, depth=1,
                          normalize_median=normalize_median)
    ok, lam = _gp.is_positive_semidefinite(K) if Z is None \
        else (None, None)
    res = RichResult(payload={"estimate": K[0][0], "kernel": K,
                              "positive_semidefinite": ok,
                              "eigenvalues": lam,
                              "method": "arc-cosine kernel, one hidden layer (MVSML 2022 eq. 8.4)"})
    return with_describe_pointer(res, "msm131")


mvsml_arccos_kernel = mvsml_categorical_count_eq_8_4


def cheatsheet():
    return "msm131: Arc-cosine kernel with one hidden layer"
