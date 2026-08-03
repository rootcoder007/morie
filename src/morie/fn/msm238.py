# morie.fn -- function file (rootcoder007/morie)
"""One-way fixed-effects model.

Implements eq. (1.3) p.16 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_general_eq_1_3"]


def mvsml_general_eq_1_3(groups):
    """GY_ij = beta_i + e_ij (eq. 1.3): a separate fixed effect per
    level. With Table 1.1 the book reports 7.396, 6.999, 6.255, 5.543,
    5.869 and a residual standard error of 0.095. Keys: estimate."""
    s = _gp.one_way_summary(groups)
    res = RichResult(payload={"estimate": s["group_means"][0],
                              "beta": s["group_means"],
                              "sd_residual": s["sd_residual"],
                              "method": "one-way fixed effects (MVSML 2022 eq. 1.3)"})
    return with_describe_pointer(res, "msm238")


def cheatsheet():
    return "msm238: One-way fixed-effects model"
