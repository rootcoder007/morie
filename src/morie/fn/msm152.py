# morie.fn -- function file (rootcoder007/morie)
"""Compressed kernel design via the Nystrom approximation.

Implements eq. (8.12) p.291 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries the previous chapter's topic label;
chapter 8 is Reproducing Kernel Hilbert Spaces regression, and the
canonical name below reflects that.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_categorical_count_eq_8_12", "mvsml_sparse_kernel_design"]


def mvsml_categorical_count_eq_8_12(X, m_index, kernel="linear", gamma=None):
    """y = mu 1_n + P f + eps with f ~ N(0, sigma2_f I_m) and
    P = K_{n,m} U S^(-1/2) (eq. 8.12), the compressed kernel of
    Cuevas et al. (2020).  Steps 1-5 of p.291: form K_{m,m} from m
    training lines, form K_{n,m} against all n, take the
    eigendecomposition of K_{m,m}, build the design, and fit under a
    ridge framework.  P P' reproduces the Nystrom approximation
    Q = K_{n,m} K_{m,m}^-1 K_{n,m}', so only m effects are
    estimated and projected into the n-dimensional space.
    Keys: estimate."""
    f = _gp.sparse_kernel_design(X, m_index, kernel=kernel,
                                 gamma=gamma)
    res = RichResult(payload={"estimate": float(f["rank"]),
                              "P": f["P"], "Q": f["Q"],
                              "rank": f["rank"],
                              "K_mm": f["K_mm"], "K_nm": f["K_nm"],
                              "method": "compressed kernel design (MVSML 2022 eq. 8.12)"})
    return with_describe_pointer(res, "msm152")


mvsml_sparse_kernel_design = mvsml_categorical_count_eq_8_12


def cheatsheet():
    return "msm152: Compressed kernel design via the Nystrom approximation"
