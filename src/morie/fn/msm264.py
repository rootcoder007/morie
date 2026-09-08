# morie.fn -- function file (rootcoder007/morie)
"""Functional model reduced to a scalar regression.

Implements eq. (14.3) p.580 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter; the
canonical name below reflects the chapter this equation is actually in.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_convolutional_nn_eq_14_3", "mvsml_fda_scalar_form"]


def mvsml_convolutional_nn_eq_14_3(t, X_curves, L1=3, L2=5, kind="fourier"):
    """Y = mu + sum_l beta_l int_0^T x(t) phi_l(t) dt + eps
    = x*'beta + eps (eq. 14.3): expanding beta(t) on L1 basis
    functions turns the functional model into an ordinary linear one
    whose covariates are x_l = int x(t) phi_l(t) dt.
    Keys: estimate."""
    d = _gp.fda_design_matrix(t, X_curves, L1, L2, kind=kind)
    res = RichResult(payload={"estimate": d["X_star"][0][0],
                              "X": d["X"], "X_star": d["X_star"],
                              "Q": d["Q"],
                              "method": "functional model in scalar form (MVSML 2022 eq. 14.3)"})
    return with_describe_pointer(res, "msm264")


mvsml_fda_scalar_form = mvsml_convolutional_nn_eq_14_3


def cheatsheet():
    return "msm264: Functional model reduced to a scalar regression"
