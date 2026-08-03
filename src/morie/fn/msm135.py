# morie.fn -- function file (rootcoder007/morie)
"""RKHS estimating equations.

Implements eq. (8.6) p.276 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries the previous chapter's topic label;
chapter 8 is Reproducing Kernel Hilbert Spaces regression, and the
canonical name below reflects that.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_categorical_count_eq_8_6", "mvsml_rkhs_mixed_equations"]


def mvsml_categorical_count_eq_8_6(C, K, y, lam=1.0, sigma2_e=1.0):
    """[C'C, C'K; K'C, K'K + lambda K sigma2_e][theta; beta]
    = [C'y; K'y] (eq. 8.6): the stationarity conditions of the
    penalized criterion J[theta, beta | lambda] under the linear mixed
    model, with C the fixed-effects design and K a valid (symmetric,
    positive semi-definite) kernel.  sigma2_beta = 1/lambda reads as
    the variation due to marked additive genomic variation.
    Keys: estimate."""
    f = _gp.rkhs_mixed_equations(C, K, y, lam=lam,
                                 sigma2_e=sigma2_e, form="direct")
    res = RichResult(payload={"estimate": f["theta"][0],
                              "theta": f["theta"],
                              "beta": f["beta"], "u": f["u"],
                              "fitted": f["fitted"],
                              "sigma2_beta": f["sigma2_beta"],
                              "method": "RKHS estimating equations (MVSML 2022 eq. 8.6)"})
    return with_describe_pointer(res, "msm135")


mvsml_rkhs_mixed_equations = mvsml_categorical_count_eq_8_6


def cheatsheet():
    return "msm135: RKHS estimating equations"
