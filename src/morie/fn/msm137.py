# morie.fn -- function file (rootcoder007/morie)
"""Reduced RKHS estimating equations.

Implements eq. (8.7) p.276 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries the previous chapter's topic label;
chapter 8 is Reproducing Kernel Hilbert Spaces regression, and the
canonical name below reflects that.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_categorical_count_eq_8_7", "mvsml_rkhs_mixed_equations_reduced"]


def mvsml_categorical_count_eq_8_7(C, K, y, lam=1.0, sigma2_e=1.0, K_star=None):
    """[C'C, C'K; I'C, K + lambda I sigma2_e][theta; beta]
    = [C'y; y] (eq. 8.7): multiplying the second block of eq. (8.6) by
    K^-1 removes both the inverse of K and the product K'K, which
    matters because K can reach 100,000 x 100,000.  The book states the
    two parameterizations give the same solution.  With ``K_star`` the
    breeding values of new genotyped individuals follow from the single
    product u_new = K_s beta. Keys: estimate."""
    f = _gp.rkhs_mixed_equations(C, K, y, lam=lam,
                                 sigma2_e=sigma2_e, form="reduced")
    u_new = _gp.rkhs_predict_new(K_star, f["beta"]) \
        if K_star is not None else None
    res = RichResult(payload={"estimate": f["theta"][0],
                              "theta": f["theta"],
                              "beta": f["beta"], "u": f["u"],
                              "fitted": f["fitted"],
                              "u_new": u_new,
                              "method": "reduced RKHS estimating equations (MVSML 2022 eq. 8.7)"})
    return with_describe_pointer(res, "msm137")


mvsml_rkhs_mixed_equations_reduced = mvsml_categorical_count_eq_8_7


def cheatsheet():
    return "msm137: Reduced RKHS estimating equations"
