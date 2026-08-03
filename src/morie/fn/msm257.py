# morie.fn -- function file (rootcoder007/morie)
"""Model comparison across the chapter-1 fits.

Implements eq. (1.2)-(1.5) pp.15-16 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_general_eq_1_222"]


def mvsml_general_eq_1_222(groups):
    """Compares the single-mean fit (1.2), the fixed-effects fit (1.3)
    and the random-effects fit (1.5) on the same layout: the residual
    standard error collapses from 0.7197 to 0.095 for the Table 1.1
    data, which is what the book uses to argue that the environment
    effect matters. Keys: estimate."""
    s = _gp.one_way_summary(groups)
    ratio = s["sd_single_mean"] / s["sd_residual"] \
        if s["sd_residual"] > 0 else float("inf")
    res = RichResult(payload={"estimate": ratio,
                              "sd_single_mean": s["sd_single_mean"],
                              "sd_residual": s["sd_residual"],
                              "sigma2_b": s["sigma2_b"],
                              "method": "chapter-1 model comparison (MVSML 2022 eq. 1.2-1.5)"})
    return with_describe_pointer(res, "msm257")


def cheatsheet():
    return "msm257: Model comparison across the chapter-1 fits"
