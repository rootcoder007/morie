# morie.fn -- function file (rootcoder007/morie)
"""Multi-trait genomic linear mixed model.

Implements eq. (5.5) p.153 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_linear_mixed_models_eq_5_5"]


def mvsml_linear_mixed_models_eq_5_5(Y, Z, G, Sigma_T, R_T):
    """Stacking the n_T traits of each line, Y = (1 (x) I_nT) mu
    + Z b + eps with b ~ N(0, G (x) Sigma_T) and eps ~ N(0, I_J (x)
    R_nT) (eq. 5.5).  Sigma_T is the genetic covariance between
    traits.  When Sigma_T and R are diagonal the fit is equivalent to
    fitting a univariate GBLUP per trait. Keys: estimate."""
    f = _gp.multitrait_model(Y, Z, G, Sigma_T, R_T)
    res = RichResult(payload={"estimate": f["mu"][0], "mu": f["mu"],
                              "b": f["b"],
                              "b_by_line": f["b_by_line"],
                              "method": "multi-trait genomic LMM (MVSML 2022 eq. 5.5)"})
    return with_describe_pointer(res, "msm031")


def cheatsheet():
    return "msm031: Multi-trait genomic linear mixed model"
