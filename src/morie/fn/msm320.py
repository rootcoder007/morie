# morie.fn -- function file (rootcoder007/morie)
"""Single-mean model for a one-way layout.

Implements eq. (1.2) p.15 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_general_eq_1_2"]


def mvsml_general_eq_1_2(groups):
    """GY_ij = beta + e_ij (eq. 1.2): one grand mean for all levels.
    With Table 1.1 the book reports beta-hat = 6.4127 and a residual
    standard error of 0.7197. Keys: estimate."""
    s = _gp.one_way_summary(groups)
    res = RichResult(payload={"estimate": s["grand_mean"],
                              "beta": s["grand_mean"],
                              "sd_residual": s["sd_single_mean"],
                              "method": "single-mean model (MVSML 2022 eq. 1.2)"})
    return with_describe_pointer(res, "msm320")


def cheatsheet():
    return "msm320: Single-mean model for a one-way layout"
