# morie.fn -- function file (rootcoder007/morie)
"""Brier score for categorical data.

Implements eq. (4.14) p.136 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_overfitting_resampling_eq_4_14"]


def mvsml_overfitting_resampling_eq_4_14(probs, y_true, n_classes=None, halved=False):
    """BS = T^-1 sum_i sum_c (pi_ic - d_ic)^2 (eq. 4.14) with d_ic the
    indicator of the observed category. The categorical score lies in
    [0, 2]; halving it puts it in [0, 1] as the book suggests. Lower is
    better. Keys: estimate."""
    bs = _gp.brier_score(probs, y_true, n_classes, halved=halved)
    mll = _gp.mean_log_loss(probs, y_true, n_classes)
    res = RichResult(payload={"estimate": bs, "brier": bs,
                              "mean_log_loss": mll,
                              "method": "Brier score (MVSML 2022 eq. 4.14)"})
    return with_describe_pointer(res, "msm009")


def cheatsheet():
    return "msm009: Brier score for categorical data"
