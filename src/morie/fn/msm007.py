# morie.fn -- function file (rootcoder007/morie)
"""Generalized precision.

Implements eq. (4.9) p.132 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_overfitting_resampling_eq_4_9"]


def mvsml_overfitting_resampling_eq_4_9(y_true, y_pred, class_index=0, n_classes=None):
    """P_i = TTP_all / (TTP_all + TFP_i) on a one-versus-all basis
    (eq. 4.9), with TFP_i = sum_{j != i} n_ji from eq. (4.6).
    Keys: estimate."""
    conf = _gp.confusion_counts(y_true, y_pred, n_classes)
    m = _gp.class_metrics(conf, int(class_index))
    res = RichResult(payload={"estimate": m["precision"],
                              "TFP": m["TFP"], "TTP_all": m["TTP_all"],
                              "pCCC": m["pCCC"],
                              "method": "generalized precision (MVSML 2022 eq. 4.9)"})
    return with_describe_pointer(res, "msm007")


def cheatsheet():
    return "msm007: Generalized precision"
