# morie.fn -- function file (rootcoder007/morie)
"""Eigenvalue reparameterization of the kernel model.

Implements eq. (8.11) p.289 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries the previous chapter's topic label;
chapter 8 is Reproducing Kernel Hilbert Spaces regression, and the
canonical name below reflects that.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_categorical_count_eq_8_11", "mvsml_kernel_eigen_design"]


def mvsml_categorical_count_eq_8_11(K, tol=1e-10):
    """y = mu 1_n + P f + eps with f ~ N(0, sigma2_f I_r) and
    P = U S^(1/2) from the eigendecomposition K = U S^(1/2) S^(1/2) U'
    (eq. 8.11).  Since P P' = K this is exactly model (8.8), but
    written as a conventional ridge regression on r = rank(K)
    columns; r is usually well below min(n, p) in multi-environment
    and multi-trait models, so the reparameterization is much cheaper.
    Keys: estimate."""
    f = _gp.kernel_eigen_design(K, tol=tol)
    res = RichResult(payload={"estimate": float(f["rank"]),
                              "P": f["P"], "rank": f["rank"],
                              "eigenvalues": f["eigenvalues"],
                              "method": "eigenvalue kernel reparameterization (MVSML 2022 eq. 8.11)"})
    return with_describe_pointer(res, "msm145")


mvsml_kernel_eigen_design = mvsml_categorical_count_eq_8_11


def cheatsheet():
    return "msm145: Eigenvalue reparameterization of the kernel model"
