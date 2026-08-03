# morie.fn -- function file (rootcoder007/morie)
"""RKHS estimation in the frequentist framework.

Implements eq. (8.3) p.254 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the auto-generated stub name carries the topic label of the
previous chapter; chapter 8 is Reproducing Kernel Hilbert Spaces
regression, and the canonical name below reflects that.  Both names
resolve to the same function.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_categorical_count_eq_8_3", "mvsml_rkhs_fit"]


def mvsml_categorical_count_eq_8_3(K, y, lam=1.0):
    """min over (eta_0, beta) of {(1/n) sum_i L(y_i, eta_0 + k_i'beta)
    + (lambda/2) beta'K beta} (eq. 8.3), obtained by substituting the
    representer form (8.2) into (8.1).  beta'K beta is the empirical
    RKHS norm and lambda controls the trade-off between goodness of
    fit and complexity; with the squared-error loss the stationarity
    conditions are linear and are solved directly here.
    Keys: estimate."""
    f = _gp.rkhs_fit_squared_loss(K, y, lam=lam)
    res = RichResult(payload={"estimate": f["eta0"],
                              "eta0": f["eta0"], "beta": f["beta"],
                              "fitted": f["fitted"],
                              "objective": f["objective"],
                              "penalty": f["penalty"],
                              "method": "RKHS frequentist fit (MVSML 2022 eq. 8.3)"})
    return with_describe_pointer(res, "msm128")


mvsml_rkhs_fit = mvsml_categorical_count_eq_8_3


def cheatsheet():
    return "msm128: RKHS estimation in the frequentist framework"
