# morie.fn -- function file (rootcoder007/morie)
"""Reparameterized one-way model.

Implements eq. (1.4) p.16 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_general_eq_1_4"]


def mvsml_general_eq_1_4(groups):
    """GY_ij = beta-bar + (beta_i - beta-bar) + e_ij (eq. 1.4) with
    beta-bar = sum_i beta_i / 5: the fixed-effects fit rewritten around
    the average level, one step from the random-effects version.
    Keys: estimate."""
    s = _gp.one_way_summary(groups)
    res = RichResult(payload={"estimate": s["grand_mean"],
                              "beta_bar": s["grand_mean"],
                              "deviations": s["deviations"],
                              "deviations_sum": sum(s["deviations"]),
                              "method": "reparameterized one-way model (MVSML 2022 eq. 1.4)"})
    return with_describe_pointer(res, "msm005")


def cheatsheet():
    return "msm005: Reparameterized one-way model"
