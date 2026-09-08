# morie.fn -- function file (rootcoder007/morie)
"""One-way random-effects (mixed) model.

Implements eq. (1.5) p.16 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_general_eq_1_5"]


def mvsml_general_eq_1_5(groups):
    """GY_ij = beta + b_i + e_ij with b_i ~ N(0, sigma2_b) (eq. 1.5).
    Balanced ANOVA estimators: sigma2_e = MSE, sigma2_b =
    (MSB - MSE)/r; the correlation within a level is
    sigma2_b/(sigma2_b + sigma2). With Table 1.1 the book reports
    beta = 6.413, sigma2_b = 0.594 and sigma = 0.095. Keys: estimate."""
    s = _gp.one_way_summary(groups)
    res = RichResult(payload={"estimate": s["grand_mean"],
                              "beta": s["grand_mean"],
                              "sigma2_b": s["sigma2_b"],
                              "sd_residual": s["sd_residual"],
                              "icc": s["icc"],
                              "method": "one-way random effects (MVSML 2022 eq. 1.5)"})
    return with_describe_pointer(res, "msm006")


def cheatsheet():
    return "msm006: One-way random-effects (mixed) model"
